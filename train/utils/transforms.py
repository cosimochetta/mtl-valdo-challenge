from monai.transforms import (
    MapTransform
)

from monai.config import KeysCollection
from typing import Dict, Hashable, List, Mapping, Union
import numpy as np

class ResetBackgroundd(MapTransform):
    """
    Set values to 0 where mask is 0.
    
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        mask: str,
        allow_missing_keys: bool = False,
        
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            mask: key corrisponding to the binary mask
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys)
        self.mask = mask
        
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> List[Dict[Hashable, np.ndarray]]:
        
        results: Dict[Hashable, np.ndarray] = {}
        
        bg_mask = data[self.mask]

        for key in self.keys:
            img = data[key]
            img[bg_mask == 0] = 0
            results[key] = img
        
        # fill in the extra keys with unmodified data
        for key in set(data.keys()).difference(set(self.keys)):
            results[key] = data[key]
                
        return results

class RemoveOutliersd(MapTransform):
    """
    Transform class that sets to mean data 
    higher or lower than specified values.
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        max_value: Union[List[float], float],
        min_value: Union[List[float], float] = 0,
        mask: str = None,
        allow_missing_keys: bool = False,
        
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            min_value: set to mean values below min_value
            max_value: set to mean values greater that max_value
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys)
        if type(min_value) is not list: self.min_value = [ min_value ] * len(keys) 
        else: self.min_value = min_value
        if type(max_value) is not list: self.min_value = [ max_value ] * len(keys) 
        else: self.max_value = max_value
                
        assert len(self.max_value) == len(self.keys)
        assert len(self.min_value) == len(self.keys)
        
        

        
    def _get_masked_mean(self, data, mask, min_value, max_value):
        mask = np.logical_or(data <= min_value, data > max_value)
        ma = np.ma.array(data, mask=mask)
        return ma.mean()
        
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> List[Dict[Hashable, np.ndarray]]:
        
        results: Dict[Hashable, np.ndarray] = {}
                
        for key, min_value, max_value in zip(self.keys, self.min_value, self.max_value):
            img = data[key]
            mean = self._get_masked_mean(img, np.median(img), min_value, max_value)
            img[img > max_value] = mean
            img[img < min_value] = mean
            results[key] = img
        
        # fill in the extra keys with unmodified data
        for key in set(data.keys()).difference(set(self.keys)):
            results[key] = data[key]
                
        return results