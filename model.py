from pydantic import BaseModel

class Scores(BaseModel):
    id : int
    rms : float
    r2 : float