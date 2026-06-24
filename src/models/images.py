from pydantic import BaseModel, computed_field


class PaintColor(BaseModel):
    name: str
    r: int
    g: int
    b: int


class AchievedColor(BaseModel):
    r: int
    g: int
    b: int

    @computed_field
    @property
    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


class MixRecipe(BaseModel):
    percentages: dict[str, int]
    achieved_color: AchievedColor


class DominantColor(BaseModel):
    r: int
    g: int
    b: int

    @computed_field
    @property
    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    recipe: MixRecipe | None = None


class PaletteEntry(BaseModel):
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
    paint_by_numbers_image: str = ""
    paint_by_numbers_filled_image: str = ""
    color_palette: dict[str, PaletteEntry] = {}
