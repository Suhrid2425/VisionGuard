
import warnings

from django import forms
from PIL import Image

from .models import Inspection


class InspectionForm(forms.ModelForm):

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_IMAGE_PIXELS = 25_000_000
    MAX_IMAGE_DIMENSION = 10_000

    class Meta:
        model = Inspection
        fields = ['category', 'image']

        widgets = {
            'category': forms.Select(
                attrs={'class': 'form-select'}
            ),
            'image': forms.ClearableFileInput(
                attrs={
                    'class': 'form-control',
                    'accept': 'image/jpeg,image/png'
                }
            )
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['category'].choices = [
            ('bottle', 'Bottle'),
            ('cable', 'Cable'),
            ('capsule', 'Capsule'),
            ('carpet', 'Carpet'),
            ('grid', 'Grid'),
            ('hazelnut', 'Hazelnut'),
            ('leather', 'Leather'),
            ('metal_nut', 'Metal Nut'),
            ('pill', 'Pill'),
            ('screw', 'Screw'),
            ('tile', 'Tile'),
            ('toothbrush', 'Toothbrush'),
            ('transistor', 'Transistor'),
            ('wood', 'Wood'),
            ('zipper', 'Zipper'),
        ]

        self.fields['category'].help_text = (
            'Select the product category to run its corresponding AI model.'
        )
    def clean_image(self):
        image = self.cleaned_data.get('image')

        if not image:
            raise forms.ValidationError(
                "Please upload an image."
            )

        # 1. Check compressed file size.
        if image.size > self.MAX_FILE_SIZE:
            raise forms.ValidationError(
                "Image size must not exceed 10 MB."
            )

        try:
            # 2. Treat Pillow decompression-bomb warnings as errors.
            with warnings.catch_warnings():
                warnings.simplefilter(
                    'error',
                    Image.DecompressionBombWarning
                )

                with Image.open(image) as img:

                    # 3. Check actual image format.
                    if img.format not in ['JPEG', 'PNG']:
                        raise forms.ValidationError(
                            "Only JPG and PNG images are supported."
                        )

                    # 4. Reject animated or multi-frame images.
                    if getattr(img, 'n_frames', 1) != 1:
                        raise forms.ValidationError(
                            "Animated images are not supported."
                        )

                    width, height = img.size

                    # 5. Check image dimensions.
                    if (
                        width > self.MAX_IMAGE_DIMENSION
                        or height > self.MAX_IMAGE_DIMENSION
                    ):
                        raise forms.ValidationError(
                            "Image dimensions must not exceed "
                            "10,000 pixels on either side."
                        )

                    # 6. Check total pixel count.
                    if width * height > self.MAX_IMAGE_PIXELS:
                        raise forms.ValidationError(
                            "Image resolution is too large. "
                            "Maximum allowed is 25 megapixels."
                        )

                    # 7. Verify image integrity.
                    img.verify()

        except forms.ValidationError:
            raise

        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning
        ):
            raise forms.ValidationError(
                "Image resolution is too large."
            )

        except Exception:
            raise forms.ValidationError(
                "Invalid or corrupted image."
            )

        # 8. Reset file pointer for Django storage and inference.
        image.seek(0)

        return image

