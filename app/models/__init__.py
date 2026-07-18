"""Importing this package registers every ORM model on Base.metadata."""
from app.models.user import User, AnnotatorProfile, Client, AnnotatorApplication
from app.models.project import Project, CvatMapping
from app.models.assignment import TaskAssignment, QualityReview
from app.models.billing import ProjectQuote, Invoice
from app.models.performance import AnnotatorPerformanceSnapshot
from app.models.dataset import DatasetImage

__all__ = [
    "User",
    "AnnotatorProfile",
    "Client",
    "AnnotatorApplication",
    "Project",
    "CvatMapping",
    "TaskAssignment",
    "QualityReview",
    "ProjectQuote",
    "Invoice",
    "AnnotatorPerformanceSnapshot",
    "DatasetImage",
]
