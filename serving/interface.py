from pydantic import BaseModel
from typing import Dict, List, Literal, Optional, Union

class ChatImageCompletionRequest(BaseModel):
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    num_beams : Optional[int] = 1
