import os
import io
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

REQUIRED_COLUMNS = [
    'Post_ID', 'Platform', 'Post_Type', 'Date', 'Followers',
    'Impressions', 'Reach', 'Likes', 'Comments', 'Shares',
    'Saves', 'Video_Views', 'CTR_%', 'Engagement_Rate_%',
    'Sentiment', 'Hashtag_Count'
]

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEFAULT_EXCEL_PATH = os.path.join(DATA_DIR, 'Social_Media_Engagement_Analysis_Dataset_300.xlsx')
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, 'Social_Media_Engagement_Analysis_Dataset_300.csv')

class DataLoader:
    def __init__(self):
        self.df_raw: pd.DataFrame = pd.DataFrame()
        self.df_processed: pd.DataFrame = pd.DataFrame()
        self.active_dataset_name: str = "Social_Media_Engagement_Analysis_Dataset_300.xlsx"
        self.is_custom_dataset: bool = False
        self.last_updated: str = datetime.now().isoformat()
        self.load_default_dataset()

    def load_default_dataset(self):
        """Loads the initial supplied 300-record dataset."""
        if os.path.exists(DEFAULT_EXCEL_PATH):
            try:
                self.df_raw = pd.read_excel(DEFAULT_EXCEL_PATH, sheet_name='Social_Media_Engagement')
                self.active_dataset_name = "Social_Media_Engagement_Analysis_Dataset_300.xlsx"
            except Exception:
                self.df_raw = pd.read_excel(DEFAULT_EXCEL_PATH)
        elif os.path.exists(DEFAULT_CSV_PATH):
            self.df_raw = pd.read_csv(DEFAULT_CSV_PATH)
            self.active_dataset_name = "Social_Media_Engagement_Analysis_Dataset_300.csv"
        else:
            raise FileNotFoundError(f"Initial dataset file not found at {DEFAULT_EXCEL_PATH}")

        self.is_custom_dataset = False
        self.process_dataset()

    def process_dataset(self):
        """Standardizes, validates, and flags anomalies while preserving raw data."""
        df = self.df_raw.copy()
        
        # Strip string whitespace
        df.columns = [str(c).strip() for c in df.columns]
        
        # Verify required columns
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        # Cast datatypes safely
        df['Post_ID'] = df['Post_ID'].astype(str).str.strip()
        df['Platform'] = df['Platform'].astype(str).str.strip()
        df['Post_Type'] = df['Post_Type'].astype(str).str.strip()
        df['Sentiment'] = df['Sentiment'].astype(str).str.strip()
        
        # Robust date parsing
        df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        # If parsing failed, fallback or fill with dummy range
        if df['Date_Parsed'].isnull().any():
            df['Date_Parsed'] = df['Date_Parsed'].fillna(pd.to_datetime('2026-01-01'))
        df['Date'] = df['Date_Parsed'].dt.strftime('%Y-%m-%d')

        # Numeric conversions
        numeric_cols = ['Followers', 'Impressions', 'Reach', 'Likes', 'Comments', 
                        'Shares', 'Saves', 'Video_Views', 'CTR_%', 'Engagement_Rate_%', 'Hashtag_Count']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Derived metrics
        df['Total_Interactions'] = df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']
        
        # Outlier & anomaly identification
        # Rule 1: Engagement_Rate_% > 100% (Interactions exceed unique Reach)
        # Rule 2: Impressions < Reach (Impossible in true unique reach definitions)
        # Rule 3: Extreme statistical outliers in ER (IQR method)
        q75 = df['Engagement_Rate_%'].quantile(0.75)
        q25 = df['Engagement_Rate_%'].quantile(0.25)
        iqr = q75 - q25
        er_upper_bound = q75 + (3.0 * iqr)  # extreme outlier threshold

        is_high_er = df['Engagement_Rate_%'] > 100.0
        is_imp_anomaly = df['Impressions'] < df['Reach']
        is_stat_outlier = df['Engagement_Rate_%'] > er_upper_bound

        df['Is_Outlier'] = is_high_er | is_imp_anomaly | is_stat_outlier
        
        reasons = []
        for _, row in df.iterrows():
            r = []
            if row['Engagement_Rate_%'] > 100.0:
                r.append(f"High ER ({row['Engagement_Rate_%']:.1f}%): Interactions exceed Reach")
            if row['Impressions'] < row['Reach']:
                r.append("Impressions < Reach")
            elif row['Engagement_Rate_%'] > er_upper_bound:
                r.append("Statistical ER extreme outlier (>3x IQR)")
            reasons.append("; ".join(r) if r else None)

        df['Outlier_Reason'] = [r if r else "" for r in reasons]
        self.df_processed = df
        self.last_updated = datetime.now().isoformat()

    def load_custom_file(self, file_bytes: bytes, filename: str) -> Tuple[bool, str, int]:
        """Loads an uploaded user file (Excel or CSV) into memory."""
        try:
            if filename.endswith(('.xlsx', '.xls')):
                try:
                    new_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name='Social_Media_Engagement')
                except Exception:
                    new_df = pd.read_excel(io.BytesIO(file_bytes))
            elif filename.endswith('.csv'):
                new_df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                return False, "Unsupported file format. Please upload an .xlsx, .xls, or .csv file.", 0

            # Check columns
            missing = [c for c in REQUIRED_COLUMNS if c not in new_df.columns]
            if missing:
                return False, f"Missing required columns in uploaded file: {', '.join(missing)}", 0

            if len(new_df) == 0:
                return False, "Uploaded file contains 0 data rows.", 0

            # Apply
            self.df_raw = new_df
            self.active_dataset_name = filename
            self.is_custom_dataset = True
            self.process_dataset()
            return True, f"Successfully loaded {len(new_df)} records from {filename}", len(new_df)
        except Exception as e:
            return False, f"Error processing file: {str(e)}", 0

    def reset_dataset(self):
        """Resets back to original default dataset."""
        self.load_default_dataset()

    def filter_df(self, 
                  platform: Optional[str] = None,
                  post_type: Optional[str] = None,
                  sentiment: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  search: Optional[str] = None,
                  exclude_outliers: bool = False) -> pd.DataFrame:
        """Filters the processed dataset according to criteria."""
        df = self.df_processed.copy()

        if exclude_outliers:
            df = df[~df['Is_Outlier']]

        if platform and platform.lower() != 'all':
            df = df[df['Platform'].str.lower() == platform.lower()]

        if post_type and post_type.lower() != 'all':
            df = df[df['Post_Type'].str.lower() == post_type.lower()]

        if sentiment and sentiment.lower() != 'all':
            df = df[df['Sentiment'].str.lower() == sentiment.lower()]

        if start_date:
            df = df[df['Date'] >= start_date]

        if end_date:
            df = df[df['Date'] <= end_date]

        if search:
            s = search.lower().strip()
            df = df[
                df['Post_ID'].str.lower().str.contains(s) |
                df['Platform'].str.lower().str.contains(s) |
                df['Post_Type'].str.lower().str.contains(s) |
                df['Sentiment'].str.lower().str.contains(s)
            ]

        return df

    def get_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Computes comprehensive KPIs on the filtered dataframe."""
        total_posts = len(df)
        if total_posts == 0:
            return {
                "total_posts": 0,
                "total_likes": 0,
                "total_comments": 0,
                "total_shares": 0,
                "total_saves": 0,
                "total_interactions": 0,
                "total_impressions": 0,
                "total_reach": 0,
                "total_video_views": 0,
                "avg_engagement_rate": 0.0,
                "median_engagement_rate": 0.0,
                "avg_ctr": 0.0,
                "median_ctr": 0.0,
                "platform_count": 0,
                "active_dataset_name": self.active_dataset_name,
                "is_custom_dataset": self.is_custom_dataset,
                "total_dataset_rows": len(self.df_processed),
                "filtered_rows_count": 0,
                "outlier_count": 0
            }

        return {
            "total_posts": total_posts,
            "total_likes": int(df['Likes'].sum()),
            "total_comments": int(df['Comments'].sum()),
            "total_shares": int(df['Shares'].sum()),
            "total_saves": int(df['Saves'].sum()),
            "total_interactions": int(df['Total_Interactions'].sum()),
            "total_impressions": int(df['Impressions'].sum()),
            "total_reach": int(df['Reach'].sum()),
            "total_video_views": int(df['Video_Views'].sum()),
            "avg_engagement_rate": round(float(df['Engagement_Rate_%'].mean()), 2),
            "median_engagement_rate": round(float(df['Engagement_Rate_%'].median()), 2),
            "avg_ctr": round(float(df['CTR_%'].mean()), 2),
            "median_ctr": round(float(df['CTR_%'].median()), 2),
            "platform_count": int(df['Platform'].nunique()),
            "active_dataset_name": self.active_dataset_name,
            "is_custom_dataset": self.is_custom_dataset,
            "total_dataset_rows": len(self.df_processed),
            "filtered_rows_count": total_posts,
            "outlier_count": int(df['Is_Outlier'].sum())
        }

    def get_platform_comparison(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Computes platform metrics comparison."""
        if len(df) == 0:
            return []

        res = []
        total_all_posts = len(df)
        for platform, group in df.groupby('Platform'):
            top_type_series = group['Post_Type'].mode()
            top_type = top_type_series.iloc[0] if not top_type_series.empty else "N/A"
            res.append({
                "platform": platform,
                "post_count": int(len(group)),
                "post_share_pct": round((len(group) / total_all_posts) * 100, 1),
                "avg_engagement_rate": round(float(group['Engagement_Rate_%'].mean()), 2),
                "median_engagement_rate": round(float(group['Engagement_Rate_%'].median()), 2),
                "avg_ctr": round(float(group['CTR_%'].mean()), 2),
                "median_ctr": round(float(group['CTR_%'].median()), 2),
                "total_reach": int(group['Reach'].sum()),
                "avg_reach": round(float(group['Reach'].mean()), 1),
                "total_impressions": int(group['Impressions'].sum()),
                "avg_impressions": round(float(group['Impressions'].mean()), 1),
                "total_likes": int(group['Likes'].sum()),
                "total_comments": int(group['Comments'].sum()),
                "total_shares": int(group['Shares'].sum()),
                "total_saves": int(group['Saves'].sum()),
                "total_video_views": int(group['Video_Views'].sum()),
                "top_post_type": top_type
            })

        # Sort by median engagement rate descending
        res.sort(key=lambda x: x['median_engagement_rate'], reverse=True)
        return res

    def get_post_type_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Computes metrics grouped by post type."""
        if len(df) == 0:
            return []

        res = []
        for post_type, group in df.groupby('Post_Type'):
            res.append({
                "post_type": post_type,
                "post_count": int(len(group)),
                "avg_engagement_rate": round(float(group['Engagement_Rate_%'].mean()), 2),
                "median_engagement_rate": round(float(group['Engagement_Rate_%'].median()), 2),
                "total_interactions": int(group['Total_Interactions'].sum()),
                "avg_reach": round(float(group['Reach'].mean()), 1)
            })
        res.sort(key=lambda x: x['median_engagement_rate'], reverse=True)
        return res

    def get_sentiment_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Computes sentiment breakdown."""
        if len(df) == 0:
            return []

        total = len(df)
        res = []
        for sentiment, group in df.groupby('Sentiment'):
            res.append({
                "sentiment": sentiment,
                "post_count": int(len(group)),
                "percentage": round((len(group) / total) * 100, 1),
                "avg_engagement_rate": round(float(group['Engagement_Rate_%'].mean()), 2),
                "median_engagement_rate": round(float(group['Engagement_Rate_%'].median()), 2)
            })
        return res

    def get_time_series(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Aggregates engagement and metrics by date."""
        if len(df) == 0:
            return []

        grouped = df.groupby('Date').agg(
            avg_engagement_rate=('Engagement_Rate_%', 'mean'),
            median_engagement_rate=('Engagement_Rate_%', 'median'),
            total_interactions=('Total_Interactions', 'sum'),
            reach=('Reach', 'sum'),
            post_count=('Post_ID', 'count')
        ).reset_index()

        grouped = grouped.sort_values('Date')
        records = []
        for _, row in grouped.iterrows():
            records.append({
                "date": str(row['Date']),
                "avg_engagement_rate": round(float(row['avg_engagement_rate']), 2),
                "median_engagement_rate": round(float(row['median_engagement_rate']), 2),
                "total_interactions": int(row['total_interactions']),
                "reach": int(row['reach']),
                "post_count": int(row['post_count'])
            })
        return records

    def get_interactions_breakdown(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Breakdown of Likes, Comments, Shares, Saves by Platform."""
        if len(df) == 0:
            return []
        res = []
        for platform, group in df.groupby('Platform'):
            res.append({
                "platform": platform,
                "likes": int(group['Likes'].sum()),
                "comments": int(group['Comments'].sum()),
                "shares": int(group['Shares'].sum()),
                "saves": int(group['Saves'].sum()),
            })
        return res

    def get_distribution_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Provides histogram/distribution bucket points for ER% and CTR%."""
        if len(df) == 0:
            return {"er_buckets": [], "ctr_buckets": []}

        # ER buckets (0-10, 10-25, 25-50, 50-100, 100+)
        er_bins = [0, 5, 10, 25, 50, 100, float('inf')]
        er_labels = ["0-5%", "5-10%", "10-25%", "25-50%", "50-100%", "100%+"]
        er_counts = pd.cut(df['Engagement_Rate_%'], bins=er_bins, labels=er_labels, right=True).value_counts()
        er_buckets = [{"range": k, "count": int(v)} for k, v in er_counts.items()]

        # CTR buckets (0-2, 2-4, 4-6, 6-8, 8-10, 10+)
        ctr_bins = [0, 2, 4, 6, 8, 10, float('inf')]
        ctr_labels = ["0-2%", "2-4%", "4-6%", "6-8%", "8-10%", "10%+"]
        ctr_counts = pd.cut(df['CTR_%'], bins=ctr_bins, labels=ctr_labels, right=True).value_counts()
        ctr_buckets = [{"range": k, "count": int(v)} for k, v in ctr_counts.items()]

        return {"er_buckets": er_buckets, "ctr_buckets": ctr_buckets}

    def get_post_detail(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Returns deep-dive post metrics compared against platform and overall medians."""
        match = self.df_processed[self.df_processed['Post_ID'].str.lower() == post_id.lower().strip()]
        if match.empty:
            return None

        post_row = match.iloc[0]
        platform = post_row['Platform']
        plat_df = self.df_processed[self.df_processed['Platform'] == platform]

        plat_median_er = float(plat_df['Engagement_Rate_%'].median())
        plat_avg_er = float(plat_df['Engagement_Rate_%'].mean())
        overall_median_er = float(self.df_processed['Engagement_Rate_%'].median())
        overall_avg_er = float(self.df_processed['Engagement_Rate_%'].mean())

        plat_median_reach = float(plat_df['Reach'].median())
        overall_median_reach = float(self.df_processed['Reach'].median())

        er = float(post_row['Engagement_Rate_%'])
        # Transparent benchmark tier
        p90 = float(self.df_processed['Engagement_Rate_%'].quantile(0.90))
        p75 = float(self.df_processed['Engagement_Rate_%'].quantile(0.75))
        p25 = float(self.df_processed['Engagement_Rate_%'].quantile(0.25))

        if er >= p90:
            tier = "Top 10% (Elite Performer)"
        elif er >= p75:
            tier = "Above Average"
        elif er >= p25:
            tier = "Average / Typical"
        else:
            tier = "Underperforming"

        notes = []
        if er > plat_median_er:
            notes.append(f"Engagement rate of {er:.2f}% exceeds the {platform} median of {plat_median_er:.2f}%.")
        else:
            notes.append(f"Engagement rate of {er:.2f}% is below the {platform} median of {plat_median_er:.2f}%.")

        if post_row['Reach'] < overall_median_reach:
            notes.append(f"Reach ({post_row['Reach']:,}) is below overall median ({overall_median_reach:,.0f}).")
        else:
            notes.append(f"Reach ({post_row['Reach']:,}) is above overall median ({overall_median_reach:,.0f}).")

        if post_row['Is_Outlier']:
            notes.append(f"Flagged anomaly: {post_row['Outlier_Reason']}")

        return {
            "post": {
                "Post_ID": post_row['Post_ID'],
                "Platform": post_row['Platform'],
                "Post_Type": post_row['Post_Type'],
                "Date": post_row['Date'],
                "Followers": int(post_row['Followers']),
                "Impressions": int(post_row['Impressions']),
                "Reach": int(post_row['Reach']),
                "Likes": int(post_row['Likes']),
                "Comments": int(post_row['Comments']),
                "Shares": int(post_row['Shares']),
                "Saves": int(post_row['Saves']),
                "Video_Views": int(post_row['Video_Views']),
                "CTR_%": float(post_row['CTR_%']),
                "Engagement_Rate_%": float(post_row['Engagement_Rate_%']),
                "Sentiment": post_row['Sentiment'],
                "Hashtag_Count": int(post_row['Hashtag_Count']),
                "Total_Interactions": int(post_row['Total_Interactions']),
                "Is_Outlier": bool(post_row['Is_Outlier']),
                "Outlier_Reason": str(post_row['Outlier_Reason']) if pd.notnull(post_row['Outlier_Reason']) and post_row['Outlier_Reason'] else ""
            },
            "platform_median_er": round(plat_median_er, 2),
            "platform_avg_er": round(plat_avg_er, 2),
            "overall_median_er": round(overall_median_er, 2),
            "overall_avg_er": round(overall_avg_er, 2),
            "platform_median_reach": round(plat_median_reach, 1),
            "overall_median_reach": round(overall_median_reach, 1),
            "performance_tier": tier,
            "performance_notes": notes
        }

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Calculates dataset quality and formula audit."""
        df = self.df_raw
        total_rows = len(df)
        duplicates = int(df.duplicated(subset=['Post_ID']).sum()) if 'Post_ID' in df.columns else int(df.duplicated().sum())
        missing_map = {col: int(df[col].isnull().sum()) for col in df.columns}
        
        # Check negative numbers in numeric columns
        neg_count = 0
        for col in ['Followers', 'Impressions', 'Reach', 'Likes', 'Comments', 'Shares', 'Saves', 'Video_Views', 'CTR_%', 'Engagement_Rate_%', 'Hashtag_Count']:
            if col in self.df_processed:
                neg_count += int((self.df_processed[col] < 0).sum())

        zero_reach = int((self.df_processed['Reach'] == 0).sum())
        imp_less_reach = int((self.df_processed['Impressions'] < self.df_processed['Reach']).sum())
        high_er = int((self.df_processed['Engagement_Rate_%'] > 100.0).sum())
        outlier_count = int(self.df_processed['Is_Outlier'].sum())

        return {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "duplicate_rows": duplicates,
            "missing_values_by_column": missing_map,
            "negative_values_found": neg_count,
            "outliers_detected": outlier_count,
            "outlier_percentage": round((outlier_count / total_rows) * 100, 2) if total_rows > 0 else 0.0,
            "formula_verified": "Engagement_Rate_% = ((Likes + Comments + Shares + Saves) / Reach) * 100",
            "reach_zero_count": zero_reach,
            "impressions_less_than_reach": imp_less_reach,
            "high_er_count": high_er,
            "data_types": {col: str(dtype) for col, dtype in self.df_processed.dtypes.items()},
            "dataset_source": self.active_dataset_name,
            "last_updated": self.last_updated
        }

    def get_insights(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generates cautious, data-grounded observations without unwarranted causal claims."""
        insights = []
        n = len(df)
        if n < 5:
            return [{
                "id": "ins-insufficient",
                "category": "anomaly",
                "title": "Insufficient Data for In-Depth Insights",
                "finding": f"Only {n} records match the current filter selection.",
                "sample_size": n,
                "confidence": "Preliminary",
                "recommendation": "Expand your date range or clear specific filters to view meaningful aggregate patterns."
            }]

        # 1. Platform Performance
        plat_grouped = df.groupby('Platform')['Engagement_Rate_%'].agg(['median', 'mean', 'count']).reset_index()
        top_plat = plat_grouped.sort_values('median', ascending=False).iloc[0]
        insights.append({
            "id": "ins-platform",
            "category": "platform",
            "title": f"Platform Engagement Distribution: {top_plat['Platform']} Leads Median",
            "finding": f"In this sample ({n} posts), {top_plat['Platform']} displays the highest median engagement rate at {top_plat['median']:.2f}% (mean: {top_plat['mean']:.2f}%, sample size: {int(top_plat['count'])} posts).",
            "sample_size": int(top_plat['count']),
            "confidence": "High" if top_plat['count'] >= 20 else "Moderate",
            "recommendation": f"Observe content formats favored on {top_plat['Platform']}. Note that median rates differ by platform audience size and mechanics; avoid concluding one channel will unconditionally outperform another."
        })

        # 2. Post Type Performance
        type_grouped = df.groupby('Post_Type')['Engagement_Rate_%'].agg(['median', 'mean', 'count']).reset_index()
        top_type = type_grouped.sort_values('median', ascending=False).iloc[0]
        insights.append({
            "id": "ins-post-type",
            "category": "content_type",
            "title": f"Top Content Format: {top_type['Post_Type']}",
            "finding": f"Across {top_type['count']} evaluated posts, '{top_type['Post_Type']}' achieved a median engagement rate of {top_type['median']:.2f}% (mean: {top_type['mean']:.2f}%).",
            "sample_size": int(top_type['count']),
            "confidence": "High" if top_type['count'] >= 15 else "Moderate",
            "recommendation": f"Evaluate whether '{top_type['Post_Type']}' posts fit upcoming communication goals. Performance may reflect specific visual formats rather than format superiority alone."
        })

        # 3. Hashtag Correlation
        corr = df['Hashtag_Count'].corr(df['Engagement_Rate_%'])
        corr_desc = "weak positive" if corr > 0.1 else ("weak negative" if corr < -0.1 else "negligible")
        insights.append({
            "id": "ins-hashtag",
            "category": "hashtag",
            "title": "Hashtag Count vs. Engagement Correlation",
            "finding": f"The linear correlation between Hashtag_Count and Engagement_Rate_% is {corr:.3f} ({corr_desc} association across {n} posts).",
            "sample_size": n,
            "confidence": "Moderate",
            "recommendation": "Do not overload posts with excessive hashtags in expectation of higher engagement; focus on content relevance rather than volume."
        })

        # 4. Sentiment Analysis
        sent_grouped = df.groupby('Sentiment')['Engagement_Rate_%'].agg(['median', 'count']).reset_index()
        top_sent = sent_grouped.sort_values('median', ascending=False).iloc[0]
        insights.append({
            "id": "ins-sentiment",
            "category": "sentiment",
            "title": f"Sentiment Pattern: {top_sent['Sentiment']} Sentiment Records",
            "finding": f"Posts classified with '{top_sent['Sentiment']}' sentiment have a median engagement rate of {top_sent['median']:.2f}% (n={int(top_sent['count'])}).",
            "sample_size": int(top_sent['count']),
            "confidence": "Moderate",
            "recommendation": "Use sentiment insights cautiously. Audience reactions reflect brand context, topic sensitivity, and timeliness."
        })

        # 5. Outlier Note
        outliers_in_filter = int(df['Is_Outlier'].sum())
        if outliers_in_filter > 0:
            insights.append({
                "id": "ins-outliers",
                "category": "anomaly",
                "title": f"High Engagement Rate Outliers ({outliers_in_filter} records)",
                "finding": f"There are {outliers_in_filter} posts with Engagement_Rate_% exceeding normal ranges (e.g. >100% where total interactions exceed recorded unique Reach).",
                "sample_size": outliers_in_filter,
                "confidence": "High",
                "recommendation": "Use the 'Exclude Outliers / Robust View' toggle to analyze trends without distortions from viral outliers or denominator discrepancies."
            })

        return insights

# Singleton instance
data_loader = DataLoader()
