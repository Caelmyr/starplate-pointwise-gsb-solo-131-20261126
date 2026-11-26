class Header(dict):
    def copy(self): return Header(self)
    def get(self, key, default=None): return super().get(key, default)
    @classmethod
    def fromstring(cls, *a, **k): return cls()
    def tostring(self, *a, **k): return ""
    @property
    def cards(self): return iter(())
class PrimaryHDU: pass
class ImageHDU: pass
class CompImageHDU: pass
class HDUList(list): pass
def open(*a, **k): raise FileNotFoundError("stub")
