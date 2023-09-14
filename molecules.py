from pydantic import BaseModel, validator
import pickle

class Molecule(BaseModel):
    smiles: str
    zinc_id: Optional[int] = None
    pubchem_id: Optional[str] = None
    eis_code: Optional[str] = None
    alc1_index: Optional[int] = None
    alc1_mean_inhibition: Optional[float] = None
    alc1_inhibition1: Optional[float] = None
    alc1_inhibition2: Optional[float] = None
    alc1_dock6: Optional[Dict[str, float]] = None
    chd1_index: Optional[int] = None
    chd1_inhibition: Optional[float] = None
    esps_data: Optional[bytes] = None
    purchasable: bool = False

    @validator('esps_data', pre=True)
    def unpickle_esps_data(cls, v):
        if v is not None:
            v = pickle.loads(v)
        return v

    # ... rest of your class ...