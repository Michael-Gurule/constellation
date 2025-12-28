"""
FastAPI application for CONSTELLATION system.
Provides REST API endpoints for ML models and system operations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.system_integrator import ConstellationSystem
from config.settings import MODELS_DIR

# Initialize FastAPI app
app = FastAPI(
    title="CONSTELLATION API",
    description="Satellite Fleet Health Management System API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
system = ConstellationSystem()

# Pydantic models
class HealthStatus(BaseModel):
    """System health status response."""
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    message: str

class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request."""
    subsystem: str = Field(default="attitude_control", description="Subsystem to analyze")
    sample_size: Optional[int] = Field(default=50000, description="Number of records to analyze")

class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response."""
    total_records: int
    anomalies_detected: int
    anomaly_rate: float
    avg_anomaly_score: float
    timestamp: datetime

class ComponentHealth(BaseModel):
    """Component health information."""
    component: str
    health_score: float
    status: str
    anomaly_score: float
    last_updated: datetime

class MaintenanceTask(BaseModel):
    """Maintenance task information."""
    task_id: str
    component: str
    start_time: datetime
    end_time: datetime
    duration_hours: float
    urgency: float
    impact: float
    risk_score: float

class AnalysisRequest(BaseModel):
    """Complete system analysis request."""
    subsystem: str = Field(default="attitude_control", description="Subsystem to analyze")
    export_results: bool = Field(default=True, description="Export results to files")

class AnalysisResponse(BaseModel):
    """Complete system analysis response."""
    status: str
    subsystem: str
    timestamp: datetime
    components_analyzed: int
    anomalies_detected: int
    alerts_generated: int
    maintenance_tasks: int
    message: str


# Health check endpoints
@app.get("/", response_model=HealthStatus)
async def root():
    """Root endpoint - health check."""
    
    models_status = {
        "isolation_forest": (MODELS_DIR / "isolation_forest.pkl").exists(),
        "fault_classifier": (MODELS_DIR / "fault_classifier.pkl").exists(),
        "lstm_forecaster": (MODELS_DIR / "degradation_forecaster_colab.pkl").exists()
    }
    
    all_loaded = all(models_status.values())
    
    return HealthStatus(
        status="healthy" if all_loaded else "degraded",
        timestamp=datetime.now(),
        models_loaded=models_status,
        message="CONSTELLATION API is running"
    )

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Detailed health check."""
    
    models_status = {
        "isolation_forest": (MODELS_DIR / "isolation_forest.pkl").exists(),
        "fault_classifier": (MODELS_DIR / "fault_classifier.pkl").exists(),
        "lstm_forecaster": (MODELS_DIR / "degradation_forecaster_colab.pkl").exists()
    }
    
    loaded_count = sum(models_status.values())
    total_count = len(models_status)
    
    status = "healthy" if loaded_count == total_count else "degraded" if loaded_count > 0 else "unavailable"
    
    return HealthStatus(
        status=status,
        timestamp=datetime.now(),
        models_loaded=models_status,
        message=f"{loaded_count}/{total_count} models loaded"
    )


# Model endpoints
@app.post("/api/v1/detect-anomalies", response_model=AnomalyDetectionResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Run anomaly detection on telemetry data.
    
    Args:
        request: Anomaly detection parameters
    
    Returns:
        Detection results
    """
    
    if system.anomaly_detector is None:
        system.load_models()
    
    if system.anomaly_detector is None:
        raise HTTPException(status_code=503, detail="Anomaly detector not available")
    
    try:
        # Load data
        df = system.load_latest_features(subsystem=request.subsystem)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No telemetry data found")
        
        # Sample if requested
        if request.sample_size and len(df) > request.sample_size:
            df = df.sample(request.sample_size, random_state=42)
        
        # Run detection
        df = system.run_anomaly_detection(df)
        
        # Calculate metrics
        anomalies = df[df['is_anomaly'] == 1]
        
        return AnomalyDetectionResponse(
            total_records=len(df),
            anomalies_detected=len(anomalies),
            anomaly_rate=len(anomalies) / len(df),
            avg_anomaly_score=df['anomaly_score'].mean(),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@app.get("/api/v1/health-scores", response_model=List[ComponentHealth])
async def get_health_scores(subsystem: str = "attitude_control"):
    """
    Get health scores for all components.
    
    Args:
        subsystem: Subsystem to query
    
    Returns:
        List of component health scores
    """
    
    try:
        # Load latest health scores
        from config.settings import PROCESSED_DATA_DIR
        import pandas as pd
        
        results_dir = PROCESSED_DATA_DIR / "analysis_results"
        health_files = list(results_dir.glob("health_scores_*.csv"))
        
        if not health_files:
            raise HTTPException(status_code=404, detail="No health scores available")
        
        latest_file = max(health_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # Convert to response models
        health_scores = []
        for _, row in df.iterrows():
            health_scores.append(ComponentHealth(
                component=row['component'],
                health_score=row['health_score'],
                status=row['status'],
                anomaly_score=row['anomaly_score'],
                last_updated=datetime.now()
            ))
        
        return health_scores
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Health scores not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve health scores: {str(e)}")


@app.get("/api/v1/maintenance-schedule", response_model=List[MaintenanceTask])
async def get_maintenance_schedule():
    """
    Get current maintenance schedule.
    
    Returns:
        List of scheduled maintenance tasks
    """
    
    try:
        from config.settings import PROCESSED_DATA_DIR
        import pandas as pd
        
        results_dir = PROCESSED_DATA_DIR / "analysis_results"
        schedule_files = list(results_dir.glob("maintenance_schedule_*.csv"))
        
        if not schedule_files:
            return []
        
        latest_file = max(schedule_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # Convert datetime columns
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Convert to response models
        tasks = []
        for _, row in df.iterrows():
            tasks.append(MaintenanceTask(
                task_id=row['task_id'],
                component=row['component'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                duration_hours=row['duration_hours'],
                urgency=row['urgency'],
                impact=row['impact'],
                risk_score=row['risk_score']
            ))
        
        return tasks
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve schedule: {str(e)}")


@app.post("/api/v1/run-analysis", response_model=AnalysisResponse)
async def run_complete_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Run complete system analysis.
    
    Args:
        request: Analysis parameters
        background_tasks: Background task manager
    
    Returns:
        Analysis summary
    """
    
    try:
        # Load models if not loaded
        if system.anomaly_detector is None:
            system.load_models()
        
        # Run analysis
        results = system.run_complete_analysis(subsystem=request.subsystem)
        
        # Export if requested
        if request.export_results:
            background_tasks.add_task(system.export_results)
        
        # Prepare response
        components_analyzed = len(results.get('health_scores', []))
        anomalies_detected = len(results.get('predictions', [])[results.get('predictions', [])['is_anomaly'] == 1]) if 'predictions' in results else 0
        alerts_generated = len(results.get('alerts', []))
        maintenance_tasks = len(results.get('maintenance_schedule', []))
        
        return AnalysisResponse(
            status="success",
            subsystem=request.subsystem,
            timestamp=datetime.now(),
            components_analyzed=components_analyzed,
            anomalies_detected=anomalies_detected,
            alerts_generated=alerts_generated,
            maintenance_tasks=maintenance_tasks,
            message="Analysis completed successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Model management endpoints
@app.post("/api/v1/models/load")
async def load_models():
    """Load ML models into memory."""
    
    try:
        system.load_models()
        
        return {
            "status": "success",
            "message": "Models loaded successfully",
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")


@app.get("/api/v1/models/status")
async def get_models_status():
    """Get status of all ML models."""
    
    models_status = {
        "isolation_forest": {
            "available": (MODELS_DIR / "isolation_forest.pkl").exists(),
            "loaded": system.anomaly_detector is not None
        },
        "fault_classifier": {
            "available": (MODELS_DIR / "fault_classifier.pkl").exists(),
            "loaded": system.fault_classifier is not None
        },
        "lstm_forecaster": {
            "available": (MODELS_DIR / "degradation_forecaster_colab.pkl").exists(),
            "loaded": False  # Not currently integrated
        }
    }
    
    return {
        "models": models_status,
        "timestamp": datetime.now()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)