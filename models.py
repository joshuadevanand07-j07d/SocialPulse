from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class PostRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Post_ID: str
    Platform: str
    Post_Type: str
    Date: str
    Followers: int
    Impressions: int
    Reach: int
    Likes: int
    Comments: int
    Shares: int
    Saves: int
    Video_Views: int
    CTR_percent: float = Field(..., alias="CTR_%")
    Engagement_Rate_percent: float = Field(..., alias="Engagement_Rate_%")
    Sentiment: str
    Hashtag_Count: int
    Total_Interactions: int
    Is_Outlier: bool = False
    Outlier_Reason: Optional[str] = ""

class FilterParams(BaseModel):
    platform: Optional[str] = None
    post_type: Optional[str] = None
    sentiment: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    search: Optional[str] = None
    exclude_outliers: bool = False

class KPISummary(BaseModel):
    total_posts: int
    total_likes: int
    total_comments: int
    total_shares: int
    total_saves: int
    total_interactions: int
    total_impressions: int
    total_reach: int
    total_video_views: int
    avg_engagement_rate: float
    median_engagement_rate: float
    avg_ctr: float
    median_ctr: float
    platform_count: int
    active_dataset_name: str
    is_custom_dataset: bool
    total_dataset_rows: int
    filtered_rows_count: int
    outlier_count: int

class PlatformStat(BaseModel):
    platform: str
    post_count: int
    post_share_pct: float
    avg_engagement_rate: float
    median_engagement_rate: float
    avg_ctr: float
    median_ctr: float
    total_reach: int
    avg_reach: float
    total_impressions: int
    avg_impressions: float
    total_likes: int
    total_comments: int
    total_shares: int
    total_saves: int
    total_video_views: int
    top_post_type: str

class PostTypeStat(BaseModel):
    post_type: str
    post_count: int
    avg_engagement_rate: float
    median_engagement_rate: float
    total_interactions: int
    avg_reach: float

class SentimentStat(BaseModel):
    sentiment: str
    post_count: int
    percentage: float
    avg_engagement_rate: float
    median_engagement_rate: float

class TimeSeriesPoint(BaseModel):
    date: str
    avg_engagement_rate: float
    total_interactions: int
    post_count: int
    reach: int

class PostDetailBenchmark(BaseModel):
    post: PostRecord
    platform_median_er: float
    platform_avg_er: float
    overall_median_er: float
    overall_avg_er: float
    platform_median_reach: float
    overall_median_reach: float
    performance_tier: str  # 'Top 10%', 'Above Average', 'Average', 'Underperforming'
    performance_notes: List[str]

class PredictionRequest(BaseModel):
    platform: str
    post_type: str
    sentiment: str = "Positive"
    followers: int = Field(..., ge=0)
    impressions: int = Field(..., ge=0)
    reach: int = Field(..., ge=0)
    video_views: int = Field(0, ge=0)
    hashtag_count: int = Field(..., ge=0, le=50)

class ModelMetricItem(BaseModel):
    model_name: str
    mae: float
    rmse: float
    r2: float
    is_baseline: bool
    cv_r2_mean: float
    cv_r2_std: float

class PredictionResponse(BaseModel):
    predicted_engagement_rate_pct: float
    estimated_engagement_count: int
    confidence_interval: Dict[str, float]
    selected_model: str
    baseline_predicted_er_pct: float
    model_r2_score: float
    model_mae: float
    reliability_warning: Optional[str] = None
    input_features: Dict[str, Any]
    formula_basis: str

class ModelMetricsResponse(BaseModel):
    train_size: int
    test_size: int
    target_variable: str
    feature_names: List[str]
    models: List[ModelMetricItem]
    best_model_name: str
    feature_importances: Dict[str, float]
    dataset_limitation_note: str

class DataQualityReport(BaseModel):
    total_rows: int
    total_columns: int
    duplicate_rows: int
    missing_values_by_column: Dict[str, int]
    negative_values_found: int
    outliers_detected: int
    outlier_percentage: float
    formula_verified: str
    reach_zero_count: int
    impressions_less_than_reach: int
    high_er_count: int
    data_types: Dict[str, str]
    dataset_source: str
    last_updated: str

class InsightItem(BaseModel):
    id: str
    category: str
    title: str
    finding: str
    sample_size: int
    confidence: str
    recommendation: str
