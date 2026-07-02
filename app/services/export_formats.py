"""Client-facing delivery formats → CVAT export format strings.

The client picks their format when they create the project (before any work
starts), so delivery never stalls on a "what format do you need?" email.
"""
from __future__ import annotations

from app.models.project import DeliveryFormat

CVAT_EXPORT_FORMATS: dict[str, str] = {
    DeliveryFormat.coco.value: "COCO 1.0",
    DeliveryFormat.yolo.value: "YOLO 1.1",
    DeliveryFormat.voc.value: "PASCAL VOC 1.1",
    DeliveryFormat.cvat_xml.value: "CVAT for images 1.1",
    DeliveryFormat.datumaro.value: "Datumaro 1.0",
}

FORMAT_LABELS: dict[str, str] = {
    DeliveryFormat.coco.value: "COCO JSON",
    DeliveryFormat.yolo.value: "YOLO TXT",
    DeliveryFormat.voc.value: "Pascal VOC XML",
    DeliveryFormat.cvat_xml.value: "CVAT XML",
    DeliveryFormat.datumaro.value: "Datumaro JSON",
}


def cvat_format_for(delivery_format: str) -> str:
    return CVAT_EXPORT_FORMATS.get(
        delivery_format, CVAT_EXPORT_FORMATS[DeliveryFormat.coco.value]
    )
