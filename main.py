import os
import io
from datetime import datetime
from fastapi import FastAPI, Query, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional, List, Dict, Any

from models import (
    KPISummary, PostRecord, PostDetailBenchmark, PredictionRequest,
    PredictionResponse, ModelMetricsResponse, DataQualityReport, InsightItem
)
from data_loader import data_loader, REQUIRED_COLUMNS
from ml_pipeline import ml_pipeline
from report_generator import generate_pdf_report

app = FastAPI(
    title="SocialPulse AI API",
    description="Backend API for Social Media Engagement Analytics & Machine Learning Prediction",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML pipeline on startup
@app.on_event("startup")
def startup_event():
    try:
        ml_pipeline.train_and_evaluate(data_loader.df_processed)
        print("ML Pipeline successfully trained on startup.")
    except Exception as e:
        print(f"Warning: ML Pipeline training on startup failed: {e}")

@app.get("/api/health")
def health_check():
    if not ml_pipeline.is_trained:
        try:
            ml_pipeline.train_and_evaluate(data_loader.df_processed)
        except Exception:
            pass
    return {
        "status": "healthy",
        "app": "SocialPulse AI",
        "active_dataset": data_loader.active_dataset_name,
        "is_custom_dataset": data_loader.is_custom_dataset,
        "total_rows": len(data_loader.df_processed),
        "ml_model_ready": ml_pipeline.is_trained
    }

@app.get("/api/dataset-info")
def get_dataset_info():
    return {
        "active_dataset_name": data_loader.active_dataset_name,
        "is_custom_dataset": data_loader.is_custom_dataset,
        "total_records": len(data_loader.df_processed),
        "columns": list(data_loader.df_processed.columns),
        "date_range": {
            "min": str(data_loader.df_processed['Date'].min()),
            "max": str(data_loader.df_processed['Date'].max())
        },
        "platforms": sorted(data_loader.df_processed['Platform'].unique().tolist()),
        "post_types": sorted(data_loader.df_processed['Post_Type'].unique().tolist()),
        "sentiments": sorted(data_loader.df_processed['Sentiment'].unique().tolist()),
        "last_updated": data_loader.last_updated
    }

@app.get("/api/summary", response_model=Dict[str, Any])
def get_summary(
    platform: Optional[str] = Query(None),
    post_type: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    exclude_outliers: bool = Query(False)
):
    df_filtered = data_loader.filter_df(
        platform=platform,
        post_type=post_type,
        sentiment=sentiment,
        start_date=start_date,
        end_date=end_date,
        search=search,
        exclude_outliers=exclude_outliers
    )

    kpis = data_loader.get_kpis(df_filtered)
    time_series = data_loader.get_time_series(df_filtered)
    platforms = data_loader.get_platform_comparison(df_filtered)
    post_types = data_loader.get_post_type_stats(df_filtered)
    sentiments = data_loader.get_sentiment_stats(df_filtered)
    interactions = data_loader.get_interactions_breakdown(df_filtered)
    distributions = data_loader.get_distribution_data(df_filtered)

    # Top 5 performing posts in filter
    top_posts_df = df_filtered.sort_values('Engagement_Rate_%', ascending=False).head(5)
    top_posts = top_posts_df.fillna("").to_dict(orient='records')

    return {
        "kpis": kpis,
        "time_series": time_series,
        "platforms": platforms,
        "post_types": post_types,
        "sentiments": sentiments,
        "interactions_breakdown": interactions,
        "distributions": distributions,
        "top_posts": top_posts
    }

@app.get("/api/posts")
def get_posts(
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=100),
    sort_by: str = Query("Date"),
    sort_desc: bool = Query(True),
    platform: Optional[str] = Query(None),
    post_type: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    exclude_outliers: bool = Query(False)
):
    df = data_loader.filter_df(
        platform=platform,
        post_type=post_type,
        sentiment=sentiment,
        start_date=start_date,
        end_date=end_date,
        search=search,
        exclude_outliers=exclude_outliers
    )

    total_count = len(df)

    # Sorting
    sort_column = sort_by if sort_by in df.columns else "Date"
    df_sorted = df.sort_values(sort_column, ascending=not sort_desc)

    # Pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = df_sorted.iloc[start_idx:end_idx].fillna("").to_dict(orient="records")

    total_pages = max(1, (total_count + page_size - 1) // page_size)

    return {
        "items": page_items,
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }

@app.get("/api/posts/{post_id}", response_model=PostDetailBenchmark)
def get_post_detail(post_id: str):
    detail = data_loader.get_post_detail(post_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Post with ID '{post_id}' not found.")
    return detail

@app.get("/api/platform-comparison")
def get_platform_comparison(
    post_type: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    exclude_outliers: bool = Query(False)
):
    df = data_loader.filter_df(
        post_type=post_type,
        sentiment=sentiment,
        start_date=start_date,
        end_date=end_date,
        exclude_outliers=exclude_outliers
    )
    platforms = data_loader.get_platform_comparison(df)
    
    # Generate concise data-driven comparison summary
    summary_notes = []
    if len(platforms) > 1:
        best_er = platforms[0]
        worst_er = platforms[-1]
        summary_notes.append(
            f"Across {len(df)} posts, {best_er['platform']} had the highest median engagement rate ({best_er['median_engagement_rate']:.2f}%), "
            f"while {worst_er['platform']} recorded the lowest ({worst_er['median_engagement_rate']:.2f}%)."
        )
        # Reach comparison
        most_reach = max(platforms, key=lambda x: x['total_reach'])
        summary_notes.append(
            f"{most_reach['platform']} accumulated the greatest total reach ({most_reach['total_reach']:,}), "
            f"accounting for {most_reach['post_share_pct']}% of analyzed posts."
        )
    return {
        "platforms": platforms,
        "total_posts": len(df),
        "summary_notes": summary_notes
    }

@app.get("/api/insights")
def get_insights(
    platform: Optional[str] = Query(None),
    post_type: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    exclude_outliers: bool = Query(False)
):
    df = data_loader.filter_df(
        platform=platform,
        post_type=post_type,
        sentiment=sentiment,
        start_date=start_date,
        end_date=end_date,
        exclude_outliers=exclude_outliers
    )
    insights = data_loader.get_insights(df)
    return {
        "insights": insights,
        "sample_size": len(df),
        "disclaimer": "Insights reflect correlations observed in the active dataset. Correlation does not imply direct causation."
    }

@app.get("/api/data-quality", response_model=DataQualityReport)
def get_data_quality():
    return data_loader.get_data_quality_report()

@app.post("/api/predict", response_model=PredictionResponse)
def predict_engagement(req: PredictionRequest):
    if not ml_pipeline.is_trained:
        ml_pipeline.train_and_evaluate(data_loader.df_processed)
    try:
        res = ml_pipeline.predict(req.model_dump())
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/model-metrics", response_model=ModelMetricsResponse)
def get_model_metrics():
    if not ml_pipeline.is_trained:
        ml_pipeline.train_and_evaluate(data_loader.df_processed)
    return ml_pipeline.get_metrics_report()

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.xlsx', '.xls', '.csv']:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload an Excel (.xlsx, .xls) or CSV (.csv) file."
        )

    # Read bytes (max 15MB limit)
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds the maximum 15MB limit.")

    success, message, row_count = data_loader.load_custom_file(contents, file.filename)
    if not success:
        raise HTTPException(status_code=400, detail=message)

    # Retrain ML pipeline on new data
    try:
        ml_pipeline.train_and_evaluate(data_loader.df_processed)
    except Exception as e:
        print(f"Retraining warning on new dataset: {e}")

    return {
        "success": True,
        "message": message,
        "total_records": row_count,
        "active_dataset": file.filename,
        "is_custom_dataset": True
    }

@app.post("/api/reset")
def reset_to_default_dataset():
    data_loader.reset_dataset()
    ml_pipeline.train_and_evaluate(data_loader.df_processed)
    return {
        "success": True,
        "message": "Reset to original 300-record dataset.",
        "active_dataset": data_loader.active_dataset_name,
        "total_records": len(data_loader.df_processed)
    }

@app.get("/api/export/csv")
def export_filtered_csv(
    platform: Optional[str] = Query(None),
    post_type: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    exclude_outliers: bool = Query(False)
):
    df = data_loader.filter_df(
        platform=platform,
        post_type=post_type,
        sentiment=sentiment,
        start_date=start_date,
        end_date=end_date,
        search=search,
        exclude_outliers=exclude_outliers
    )

    csv_data = df.to_csv(index=False)
    filename = f"socialpulse_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/export/cleaned")
def export_cleaned_csv():
    """Exports validated dataset with outliers flagged or removed."""
    df = data_loader.df_processed[~data_loader.df_processed['Is_Outlier']].copy()
    csv_data = df.to_csv(index=False)
    filename = f"socialpulse_cleaned_validated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/export/original")
def export_original_dataset():
    """Downloads original supplied Excel file or CSV."""
    excel_path = os.path.join(os.path.dirname(__file__), 'data', 'Social_Media_Engagement_Analysis_Dataset_300.xlsx')
    if os.path.exists(excel_path):
        return FileResponse(
            excel_path,
            filename="Social_Media_Engagement_Analysis_Dataset_300(1).xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'Social_Media_Engagement_Analysis_Dataset_300.csv')
    return FileResponse(csv_path, filename="Social_Media_Engagement_Analysis_Dataset_300.csv", media_type="text/csv")

@app.get("/api/export/pdf")
def export_pdf_report(
    platform: Optional[str] = Query(None),
    post_type: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    exclude_outliers: bool = Query(False)
):
    df_filtered = data_loader.filter_df(
        platform=platform,
        post_type=post_type,
        sentiment=sentiment,
        start_date=start_date,
        end_date=end_date,
        exclude_outliers=exclude_outliers
    )

    kpis = data_loader.get_kpis(df_filtered)
    platforms = data_loader.get_platform_comparison(df_filtered)
    insights = data_loader.get_insights(df_filtered)
    quality = data_loader.get_data_quality_report()
    if not ml_pipeline.is_trained:
        ml_pipeline.train_and_evaluate(data_loader.df_processed)
    ml_metrics = ml_pipeline.get_metrics_report()

    filters_applied = {
        "platform": platform,
        "post_type": post_type,
        "sentiment": sentiment,
        "start_date": start_date,
        "end_date": end_date,
        "exclude_outliers": exclude_outliers
    }

    pdf_bytes = generate_pdf_report(
        kpis=kpis,
        platforms=platforms,
        insights=insights,
        ml_metrics=ml_metrics,
        quality=quality,
        filters_applied=filters_applied
    )

    filename = f"socialpulse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
