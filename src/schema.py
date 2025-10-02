
from pydantic import BaseModel, Field

class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length (cm)")
    sepal_width: float = Field(..., ge=0, description="Sepal width (cm)")
    petal_length: float = Field(..., ge=0, description="Petal length (cm)")
    petal_width: float = Field(..., ge=0, description="Petal width (cm)")

    def to_array(self):
        # returns in the canonical iris feature order used by sklearn datasets
        return [[
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width
        ]]
