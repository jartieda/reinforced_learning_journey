from __future__ import annotations

# Re-export concrete CNN classes from shared models so this lesson can have a
# local model module learners can inspect/modify independently.
from examples.shared.models import CNNGaussianPolicy as PolicyCNN
from examples.shared.models import CNNValue as ValueCNN

__all__ = ["PolicyCNN", "ValueCNN"]
