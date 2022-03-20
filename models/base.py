from torch.nn import Module


class ZhangBase(Module):
    """Base model class containing channel normalization methods."""

    def __init__(self):
        super().__init__()

        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.


    def normalize_l(self, image_l):
        """Normalize 'L' channel."""
        return (image_l - self.l_cent)/self.l_norm


    def denormalize_l(self, image_l):
        """De-normalize 'L' channel."""
        return image_l*self.l_norm + self.l_cent


    def normalize_ab(self, image_ab):
        """Normalize 'ab' channels."""
        return image_ab/self.ab_norm


    def denormalize_ab(self, image_ab):
        """DE-normalize 'ab' channels."""
        return image_ab*self.ab_norm


    def _validate_keys(self, config, keys, argname='dict'):
        """Check if listed keys are present in dict, otherwise raise error."""
        if (not isinstance(keys, list)):
            keys = [keys]
        for key in keys:
            if (key not in config):
                raise ValueError(f'`{key}` must be in  `{argname}`')
