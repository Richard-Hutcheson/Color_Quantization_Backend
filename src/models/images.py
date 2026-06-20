from pydantic import BaseModel, computed_field


class DominantColor(BaseModel):
    r: int
    g: int
    b: int

    @computed_field
    @property
    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


class ImagesResponse(BaseModel):
    color_count: int
    colors: list[DominantColor]
