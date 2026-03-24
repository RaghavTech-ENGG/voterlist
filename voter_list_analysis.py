import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None


@dataclass
class ParseResult:
    data: pd.DataFrame
    errors: List[str]


REQUIRED_COLUMNS = [
    "district",
    "ac",
    "booth",
    "voter_name",
    "gender",
    "age",
]

SYNONYM_MAP: Dict[str, str] = {
    "district": "district",
    "dist": "district",
    "assembly constituency": "ac",
    "constituency": "ac",
    "ac": "ac",
    "part number": "booth",
    "part no": "booth",
    "booth": "booth",
    "booth number": "booth",
    "booth no": "booth",
    "serial no": "serial",
    "serial number": "serial",
    "voter name": "voter_name",
    "name": "voter_name",
    "elector name": "voter_name",
    "gender": "gender",
    "sex": "gender",
    "age": "age",
}


def normalize_column_name(col: str) -> str:
    c = re.sub(r"[\n\r\t]+", " ", str(col).strip().lower())
    c = re.sub(r"[^a-z0-9 ]+", " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    return SYNONYM_MAP.get(c, c.replace(" ", "_"))


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if "serial" not in df.columns:
        df["serial"] = np.nan
    return df


def clean_gender(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    v = str(value).strip().lower()
    if v in {"m", "male", "man"}:
        return "Male"
    if v in {"f", "female", "woman"}:
        return "Female"
    if v in {"o", "other", "third gender", "transgender"}:
        return "Other"
    return "Unknown"


def to_int_safe(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.search(r"\d{1,3}", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def normalize_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    df = ensure_required_columns(df)

    for col in ["district", "ac", "booth", "voter_name", "serial"]:
        df[col] = df[col].astype(str).str.strip().replace({"nan": ""})

    df["gender"] = df["gender"].apply(clean_gender)
    df["age"] = df["age"].apply(to_int_safe)

    mask_has_identity = (
        (df["voter_name"] != "")
        | df["age"].notna()
        | (df["gender"] != "Unknown")
    )
    df = df.loc[mask_has_identity].copy()

    df["source_file"] = source_name
    df["priority"] = ""
    df["community_profile"] = ""
    df["notes"] = ""

    return df[
        [
            "district",
            "ac",
            "booth",
            "serial",
            "voter_name",
            "gender",
            "age",
            "priority",
            "community_profile",
            "notes",
            "source_file",
        ]
    ]


def parse_csv(file_obj) -> pd.DataFrame:
    file_obj.seek(0)
    try:
        df = pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding="latin-1")
    return df


def parse_xlsx(file_obj) -> pd.DataFrame:
    file_obj.seek(0)
    xls = pd.ExcelFile(file_obj)
    frames: List[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        temp = pd.read_excel(xls, sheet_name=sheet)
        if not temp.empty:
            frames.append(temp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_pdf(file_obj) -> pd.DataFrame:
    if pdfplumber is None:
        raise RuntimeError(
            "PDF parsing dependency missing. Install with: pip install pdfplumber"
        )

    file_obj.seek(0)
    rows = []

    with pdfplumber.open(file_obj) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            district = ""
            ac = ""
            booth = ""

            for ln in lines:
                low = ln.lower()

                if "district" in low and not district:
                    district = ln.split(":")[-1].strip()

                if ("assembly constituency" in low or low.startswith("ac")) and not ac:
                    ac = ln.split(":")[-1].strip()

                if ("booth" in low or "part number" in low) and not booth:
                    booth = ln.split(":")[-1].strip()

                row_match = re.search(
                    r"name[:\-]\s*(?P<name>.+?)\s+gender[:\-]\s*(?P<gender>[A-Za-z ]+)\s+age[:\-]\s*(?P<age>\d{1,3})",
                    ln,
                    flags=re.IGNORECASE,
                )

                if row_match:
                    rows.append(
                        {
                            "district": district,
                            "ac": ac,
                            "booth": booth,
                            "voter_name": row_match.group("name").strip(),
                            "gender": row_match.group("gender").strip(),
                            "age": row_match.group("age").strip(),
                            "page_no": page_no,
                        }
                    )

    return pd.DataFrame(rows)


def parse_uploaded_file(file_obj) -> ParseResult:
    filename = file_obj.name
    ext = filename.split(".")[-1].lower()
    errors: List[str] = []

    try:
        if ext == "csv":
            raw_df = parse_csv(file_obj)
        elif ext in {"xlsx", "xls"}:
            raw_df = parse_xlsx(file_obj)
        elif ext == "pdf":
            raw_df = parse_pdf(file_obj)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if raw_df.empty:
            errors.append(f"{filename}: no readable voter records found.")
            return ParseResult(data=pd.DataFrame(), errors=errors)

        cleaned = normalize_dataframe(raw_df, source_name=filename)
        if cleaned.empty:
            errors.append(f"{filename}: parsed but no valid voter rows after cleaning.")

        return ParseResult(data=cleaned, errors=errors)

    except Exception as exc:
        errors.append(f"{filename}: parsing failed -> {exc}")
        return ParseResult(data=pd.DataFrame(), errors=errors)


@st.cache_data(show_spinner=False)
def aggregate_booth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    base = (
        df.groupby(["district", "ac", "booth"], dropna=False)
        .size()
        .reset_index(name="total_voters")
    )

    gender_counts = (
        df.pivot_table(
            index=["district", "ac", "booth"],
            columns="gender",
            values="voter_name",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    result = base.merge(gender_counts, on=["district", "ac", "booth"], how="left")

    for g in ["Male", "Female", "Other", "Unknown"]:
        if g not in result.columns:
            result[g] = 0

    return result.sort_values(["district", "ac", "booth"], na_position="last")


@st.cache_data(show_spinner=False)
def aggregate_district(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    result = (
        df.groupby(["district"], dropna=False)
        .agg(
            total_voters=("voter_name", "count"),
            male_voters=("gender", lambda x: (x == "Male").sum()),
            female_voters=("gender", lambda x: (x == "Female").sum()),
            other_voters=("gender", lambda x: (x == "Other").sum()),
            unknown_gender=("gender", lambda x: (x == "Unknown").sum()),
            avg_age=("age", "mean"),
        )
        .reset_index()
    )

    result["avg_age"] = result["avg_age"].round(1)
    return result.sort_values("district")


@st.cache_data(show_spinner=False)
def aggregate_community_profile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "community_profile" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    temp["community_profile"] = temp["community_profile"].fillna("").astype(str).str.strip()
    temp = temp[temp["community_profile"] != ""]

    if temp.empty:
        return pd.DataFrame()

    result = (
        temp.groupby(["district", "ac", "booth", "community_profile"], dropna=False)
        .size()
        .reset_index(name="tagged_voters")
        .sort_values(["district", "ac", "booth", "community_profile"])
    )
    return result


@st.cache_data(show_spinner=False)
def aggregate_district_community(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "community_profile" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    temp["community_profile"] = temp["community_profile"].fillna("").astype(str).str.strip()
    temp = temp[temp["community_profile"] != ""]

    if temp.empty:
        return pd.DataFrame()

    result = (
        temp.pivot_table(
            index="district",
            columns="community_profile",
            values="voter_name",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return result.sort_values("district")


@st.cache_data(show_spinner=False)
def get_top_10_districts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    result = (
        df.groupby("district", dropna=False)
        .agg(total_voters=("voter_name", "count"))
        .reset_index()
        .sort_values("total_voters", ascending=False)
        .head(10)
    )
    return result


@st.cache_data(show_spinner=False)
def aggregate_ac(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    result = (
        df.groupby(["district", "ac"], dropna=False)
        .agg(
            total_voters=("voter_name", "count"),
            male_voters=("gender", lambda x: (x == "Male").sum()),
            female_voters=("gender", lambda x: (x == "Female").sum()),
            other_voters=("gender", lambda x: (x == "Other").sum()),
            avg_age=("age", "mean"),
        )
        .reset_index()
    )

    result["avg_age"] = result["avg_age"].round(1)
    return result.sort_values(["district", "ac"])


@st.cache_data(show_spinner=False)
def aggregate_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "age" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    temp = temp[temp["age"].notna()].copy()

    if temp.empty:
        return pd.DataFrame()

    bins = [17, 25, 35, 45, 60, 200]
    labels = ["18-25", "26-35", "36-45", "46-60", "60+"]

    temp["age_group"] = pd.cut(temp["age"], bins=bins, labels=labels)

    result = (
        temp.groupby("age_group", dropna=False)
        .size()
        .reset_index(name="total_voters")
    )
    return result


def figure_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    districts = sorted([d for d in df["district"].dropna().unique() if str(d).strip()])
    acs = sorted([a for a in df["ac"].dropna().unique() if str(a).strip()])
    booths = sorted([b for b in df["booth"].dropna().unique() if str(b).strip()])
    genders = sorted(df["gender"].dropna().unique().tolist())

    selected_district = st.sidebar.multiselect("District", options=districts)
    selected_ac = st.sidebar.multiselect("AC", options=acs)
    selected_booth = st.sidebar.multiselect("Booth", options=booths)
    selected_gender = st.sidebar.multiselect("Gender", options=genders)

    age_series = df["age"].dropna()
    min_age = int(age_series.min()) if not age_series.empty else 18
    max_age = int(age_series.max()) if not age_series.empty else 100

    selected_age_range = st.sidebar.slider(
        "Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
    )

    temp = df.copy()

    if selected_district:
        temp = temp[temp["district"].isin(selected_district)]
    if selected_ac:
        temp = temp[temp["ac"].isin(selected_ac)]
    if selected_booth:
        temp = temp[temp["booth"].isin(selected_booth)]
    if selected_gender:
        temp = temp[temp["gender"].isin(selected_gender)]

    temp = temp[
        temp["age"].fillna(-1).between(selected_age_range[0], selected_age_range[1])
    ]

    return temp


def build_excel_download(
    booth_df: pd.DataFrame,
    district_df: pd.DataFrame,
    edited_df: pd.DataFrame,
    community_df: pd.DataFrame,
    district_community_df: pd.DataFrame,
    ac_df: pd.DataFrame,
    age_group_df: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        edited_df.to_excel(writer, index=False, sheet_name="voter_data")
        booth_df.to_excel(writer, index=False, sheet_name="booth_report")
        district_df.to_excel(writer, index=False, sheet_name="district_summary")
        community_df.to_excel(writer, index=False, sheet_name="community_summary")
        district_community_df.to_excel(writer, index=False, sheet_name="district_community")
        ac_df.to_excel(writer, index=False, sheet_name="ac_summary")
        age_group_df.to_excel(writer, index=False, sheet_name="age_groups")
    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title="Voter List Analysis Tool", layout="wide")
    st.title("Voter List Analysis Tool")
    st.caption(
        "Upload voter lists (PDF/CSV/XLSX), extract structured fields, and review booth/district analytics."
    )
    st.info(
        "This tool uses only structured data in the uploaded files and manual tagging. "
        "It does not infer religion, caste, or any other sensitive attribute."
    )

    uploaded_files = st.file_uploader(
        "Upload voter list files",
        type=["pdf", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload one or more files. Parsing quality depends on the file structure.",
    )

    if not uploaded_files:
        st.stop()

    parsed_frames: List[pd.DataFrame] = []
    all_errors: List[str] = []

    progress = st.progress(0, text="Starting file parsing...")
    total = len(uploaded_files)

    for i, file_obj in enumerate(uploaded_files, start=1):
        progress.progress(int((i - 1) / total * 100), text=f"Parsing {file_obj.name}...")
        result = parse_uploaded_file(file_obj)

        if not result.data.empty:
            parsed_frames.append(result.data)

        all_errors.extend(result.errors)
        progress.progress(int(i / total * 100), text=f"Completed {file_obj.name}")

    progress.empty()

    if all_errors:
        with st.expander("Parsing messages", expanded=True):
            for err in all_errors:
                st.error(err)

    if not parsed_frames:
        st.warning("No valid records could be extracted from uploaded files.")
        st.stop()

    full_df = pd.concat(parsed_frames, ignore_index=True)
    filtered_df = apply_filters(full_df)

    st.subheader("Search")
    search_query = st.text_input("Search by voter name, district, AC, or booth")

    if search_query:
        mask = (
            filtered_df["voter_name"].astype(str).str.contains(search_query, case=False, na=False)
            | filtered_df["district"].astype(str).str.contains(search_query, case=False, na=False)
            | filtered_df["ac"].astype(str).str.contains(search_query, case=False, na=False)
            | filtered_df["booth"].astype(str).str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Voters", int(filtered_df.shape[0]))
    c2.metric("Male", int((filtered_df["gender"] == "Male").sum()))
    c3.metric("Female", int((filtered_df["gender"] == "Female").sum()))
    c4.metric("Other", int((filtered_df["gender"] == "Other").sum()))

    c5, c6, c7 = st.columns(3)
    c5.metric("Districts", int(filtered_df["district"].nunique()))
    c6.metric("ACs", int(filtered_df["ac"].nunique()))
    c7.metric("Booths", int(filtered_df["booth"].nunique()))

    st.subheader("Manual Tagging (Editable)")
    st.caption("Edit priority, community profile, and notes. Changes apply in the current session.")

    editable_columns = [
        "district",
        "ac",
        "booth",
        "serial",
        "voter_name",
        "gender",
        "age",
        "priority",
        "community_profile",
        "notes",
    ]

    edited_df = st.data_editor(
        filtered_df[editable_columns],
        use_container_width=True,
        num_rows="dynamic",
        key="editable_voter_table",
        column_config={
            "priority": st.column_config.SelectboxColumn(
                "Priority",
                options=["", "High", "Medium", "Low"],
            ),
            "community_profile": st.column_config.SelectboxColumn(
                "Community Profile",
                options=["", "High Hindu", "Medium Hindu", "Low Hindu", "Mixed", "Review"],
            ),
            "notes": st.column_config.TextColumn("Notes"),
        },
    )

    st.subheader("Booth-wise Summary")
    booth_summary = aggregate_booth(edited_df)
    st.dataframe(booth_summary, use_container_width=True, height=320)

    st.subheader("District-wise Summary")
    district_summary = aggregate_district(edited_df)
    st.dataframe(district_summary, use_container_width=True, height=260)

    st.subheader("Top 10 District View")
    top_10_districts = get_top_10_districts(edited_df)
    if not top_10_districts.empty:
        st.dataframe(top_10_districts, use_container_width=True, height=250)
    else:
        st.info("No district data available.")

    st.subheader("Age Group Summary")
    age_group_summary = aggregate_age_groups(edited_df)
    if not age_group_summary.empty:
        st.dataframe(age_group_summary, use_container_width=True, height=220)
    else:
        st.info("No age data available.")

    if not age_group_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(age_group_summary["age_group"].astype(str), age_group_summary["total_voters"])
        ax.set_title("Age Group Distribution")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Total Voters")
        plt.tight_layout()
        st.pyplot(fig)

        st.download_button(
            "Download Age Group Chart (PNG)",
            data=figure_to_png_bytes(fig),
            file_name="age_group_distribution.png",
            mime="image/png",
        )

    st.subheader("Community Profile Summary")
    community_summary = aggregate_community_profile(edited_df)
    if not community_summary.empty:
        st.dataframe(community_summary, use_container_width=True, height=260)
    else:
        st.info("No community profile tags added yet.")

    st.subheader("District-wise Community Profile")
    district_community_summary = aggregate_district_community(edited_df)
    if not district_community_summary.empty:
        st.dataframe(district_community_summary, use_container_width=True, height=260)
    else:
        st.info("No district-wise community profile summary available yet.")

    st.subheader("District Dashboard Charts")

    if not district_summary.empty:
        chart_type = st.selectbox(
            "Select district chart",
            [
                "Total Voters by District",
                "Top 10 Districts by Total Voters",
                "Male vs Female by District",
                "Average Age by District",
            ],
        )

        if chart_type == "Total Voters by District":
            chart_df = district_summary.sort_values("total_voters", ascending=False)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(chart_df["district"], chart_df["total_voters"])
            ax.set_title("Total Voters by District")
            ax.set_xlabel("District")
            ax.set_ylabel("Total Voters")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

            st.download_button(
                "Download District Chart (PNG)",
                data=figure_to_png_bytes(fig),
                file_name="total_voters_by_district.png",
                mime="image/png",
            )

        elif chart_type == "Top 10 Districts by Total Voters":
            chart_df = district_summary.sort_values("total_voters", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(chart_df["district"], chart_df["total_voters"])
            ax.set_title("Top 10 Districts by Total Voters")
            ax.set_xlabel("District")
            ax.set_ylabel("Total Voters")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

            st.download_button(
                "Download Top 10 District Chart (PNG)",
                data=figure_to_png_bytes(fig),
                file_name="top_10_districts.png",
                mime="image/png",
            )

        elif chart_type == "Male vs Female by District":
            chart_df = district_summary.sort_values("district")

            fig, ax = plt.subplots(figsize=(12, 6))
            x = range(len(chart_df))
            width = 0.4

            ax.bar(
                [i - width / 2 for i in x],
                chart_df["male_voters"],
                width=width,
                label="Male",
            )
            ax.bar(
                [i + width / 2 for i in x],
                chart_df["female_voters"],
                width=width,
                label="Female",
            )

            ax.set_title("Male vs Female by District")
            ax.set_xlabel("District")
            ax.set_ylabel("Voters")
            ax.set_xticks(list(x))
            ax.set_xticklabels(chart_df["district"], rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            st.download_button(
                "Download Male vs Female Chart (PNG)",
                data=figure_to_png_bytes(fig),
                file_name="male_vs_female_district.png",
                mime="image/png",
            )

        elif chart_type == "Average Age by District":
            chart_df = district_summary.sort_values("avg_age", ascending=False)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(chart_df["district"], chart_df["avg_age"])
            ax.set_title("Average Age by District")
            ax.set_xlabel("District")
            ax.set_ylabel("Average Age")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

            st.download_button(
                "Download Average Age Chart (PNG)",
                data=figure_to_png_bytes(fig),
                file_name="average_age_district.png",
                mime="image/png",
            )

    if not district_community_summary.empty:
        st.subheader("Community Profile District Chart")

        community_options = [
            col for col in district_community_summary.columns if col != "district"
        ]

        if community_options:
            selected_profile = st.selectbox(
                "Select community profile chart",
                community_options,
            )

            community_chart_df = district_community_summary.sort_values(
                selected_profile, ascending=False
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(
                community_chart_df["district"],
                community_chart_df[selected_profile],
            )
            ax.set_title(f"{selected_profile} by District")
            ax.set_xlabel("District")
            ax.set_ylabel("Tagged Voters")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

            st.download_button(
                "Download Community District Chart (PNG)",
                data=figure_to_png_bytes(fig),
                file_name=f"{selected_profile.lower().replace(' ', '_')}_district.png",
                mime="image/png",
            )

    st.subheader("AC-wise Charts")
    ac_summary = aggregate_ac(edited_df)

    if not ac_summary.empty:
        ac_chart_type = st.selectbox(
            "Select AC chart",
            [
                "Top 10 ACs by Total Voters",
                "Male vs Female by AC",
                "Average Age by AC",
            ],
        )

        selected_district_for_ac = st.selectbox(
            "Select district for AC chart",
            sorted([d for d in ac_summary["district"].dropna().unique() if str(d).strip()]),
        )

        ac_chart_df = ac_summary[ac_summary["district"] == selected_district_for_ac].copy()

        if not ac_chart_df.empty:
            if ac_chart_type == "Top 10 ACs by Total Voters":
                ac_chart_df = ac_chart_df.sort_values("total_voters", ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(ac_chart_df["ac"], ac_chart_df["total_voters"])
                ax.set_title(f"Top ACs by Total Voters - {selected_district_for_ac}")
                ax.set_xlabel("AC")
                ax.set_ylabel("Total Voters")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

                st.download_button(
                    "Download AC Total Voters Chart (PNG)",
                    data=figure_to_png_bytes(fig),
                    file_name=f"ac_total_voters_{selected_district_for_ac}.png",
                    mime="image/png",
                )

            elif ac_chart_type == "Male vs Female by AC":
                ac_chart_df = ac_chart_df.sort_values("ac")

                fig, ax = plt.subplots(figsize=(12, 6))
                x = range(len(ac_chart_df))
                width = 0.4

                ax.bar(
                    [i - width / 2 for i in x],
                    ac_chart_df["male_voters"],
                    width=width,
                    label="Male",
                )
                ax.bar(
                    [i + width / 2 for i in x],
                    ac_chart_df["female_voters"],
                    width=width,
                    label="Female",
                )

                ax.set_title(f"Male vs Female by AC - {selected_district_for_ac}")
                ax.set_xlabel("AC")
                ax.set_ylabel("Voters")
                ax.set_xticks(list(x))
                ax.set_xticklabels(ac_chart_df["ac"], rotation=45, ha="right")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                st.download_button(
                    "Download AC Gender Chart (PNG)",
                    data=figure_to_png_bytes(fig),
                    file_name=f"ac_gender_{selected_district_for_ac}.png",
                    mime="image/png",
                )

            elif ac_chart_type == "Average Age by AC":
                ac_chart_df = ac_chart_df.sort_values("avg_age", ascending=False)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(ac_chart_df["ac"], ac_chart_df["avg_age"])
                ax.set_title(f"Average Age by AC - {selected_district_for_ac}")
                ax.set_xlabel("AC")
                ax.set_ylabel("Average Age")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

                st.download_button(
                    "Download AC Age Chart (PNG)",
                    data=figure_to_png_bytes(fig),
                    file_name=f"ac_age_{selected_district_for_ac}.png",
                    mime="image/png",
                )
    else:
        st.info("No AC-wise data available.")

    st.subheader("Export Reports")
    col_a, col_b = st.columns(2)

    with col_a:
        st.download_button(
            "Download Edited Voter Data (CSV)",
            data=edited_df.to_csv(index=False).encode("utf-8"),
            file_name="edited_voter_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download Booth Report (CSV)",
            data=booth_summary.to_csv(index=False).encode("utf-8"),
            file_name="booth_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download District Summary (CSV)",
            data=district_summary.to_csv(index=False).encode("utf-8"),
            file_name="district_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download Community Summary (CSV)",
            data=community_summary.to_csv(index=False).encode("utf-8"),
            file_name="community_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download District Community Summary (CSV)",
            data=district_community_summary.to_csv(index=False).encode("utf-8"),
            file_name="district_community_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download AC Summary (CSV)",
            data=ac_summary.to_csv(index=False).encode("utf-8"),
            file_name="ac_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download Age Group Summary (CSV)",
            data=age_group_summary.to_csv(index=False).encode("utf-8"),
            file_name="age_group_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_b:
        excel_bytes = build_excel_download(
            booth_summary,
            district_summary,
            edited_df,
            community_summary,
            district_community_summary,
            ac_summary,
            age_group_summary,
        )
        st.download_button(
            "Download Combined Report (Excel)",
            data=excel_bytes,
            file_name="voter_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()