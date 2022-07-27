from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int32, Int64, Bool, String

space_titanic = FileSource(
    path="/home/jupyter/data/train.parquet",
)

passenger = Entity(name="passenger", join_keys=["PassengerId"])

space_titanic_view = FeatureView(
    name="space_titanic",
    entities=[passenger],
    schema=[
        Field(name="HomePlanet", dtype=String),
        Field(name="CryoSleep", dtype=Bool),
        Field(name="Destination", dtype=String),
        Field(name="Age", dtype=Int32),
        Field(name="VIP", dtype=Bool),
        Field(name="Transported", dtype=Bool),
    ],
    online=True,
    source=space_titanic,
    tags={},
)

space_titanic_fs = FeatureService(name="space_titanic", features=[space_titanic_view])
