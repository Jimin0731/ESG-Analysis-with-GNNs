from __future__ import annotations
import os,tempfile,torch
class InMemoryBestCheckpoint:
    def __init__(self,minimum_delta=0.,path=None): self.minimum_delta=minimum_delta; self.path=path; self.state=None; self.metadata=None
    def update(self,model,value,metadata):
        if self.metadata is not None and value>=self.metadata["monitored_value"]-self.minimum_delta: return False
        self.state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; self.metadata={**metadata,"monitored_value":float(value)}
        if self.path:
            directory=os.path.dirname(os.path.abspath(self.path)); fd,tmp=tempfile.mkstemp(dir=directory); os.close(fd)
            try: torch.save({"state":self.state,"metadata":self.metadata},tmp); os.replace(tmp,self.path)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        return True
    def restore(self,model):
        if self.state is None: raise RuntimeError("no best checkpoint")
        model.load_state_dict({k:v.clone() for k,v in self.state.items()})
__all__=["InMemoryBestCheckpoint"]
