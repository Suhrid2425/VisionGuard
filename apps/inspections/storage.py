
import os
import uuid
from io import BytesIO

from django.core.files import File
from django.core.files.storage import Storage
from django.urls import reverse

from vercel import blob
from vercel.blob import BlobNotFoundError


class VercelBlobStorage(Storage):
    """
    Store inspection images in private Vercel Blob storage.
    """

    def _save(self, name, content):
        # Read uploaded or generated content.
        if hasattr(content, "chunks"):
            file_content = b"".join(content.chunks())
        else:
            file_content = content.read()

        # Upload the file to private Vercel Blob storage.
        result = blob.sync.put_blob(
            pathname=name,
            body=file_content,
            access="private",
            content_type=(
                getattr(content, "content_type", None)
                or "application/octet-stream"
            ),
            add_random_suffix=False,
        )

        return result.pathname

    def _open(self, name, mode="rb"):
        if mode not in ("rb", "r"):
            raise ValueError("Only read mode is supported.")

        result = blob.sync.get_blob(
            name,
            access="private",
        )

        return File(
            BytesIO(result.body),
            name=name,
        )

    def delete(self, name):
        if name:
            blob.sync.delete_blob(name)

    def exists(self, name):
        try:
            blob.sync.head_blob(name)
            return True
        except BlobNotFoundError:
            return False

    def get_available_name(self, name, max_length=None):
        # Generate a unique filename to avoid overwriting.
        directory, filename = os.path.split(name)
        stem, extension = os.path.splitext(filename)

        unique_suffix = f"_{uuid.uuid4().hex}"

        if max_length is not None:
            available_length = (
                max_length
                - len(directory)
                - (1 if directory else 0)
                - len(extension)
                - len(unique_suffix)
            )

            if available_length < 1:
                raise ValueError(
                    "max_length is too small for a unique filename."
                )

            stem = stem[:available_length]

        unique_name = f"{stem}{unique_suffix}{extension}"

        if directory:
            return f"{directory}/{unique_name}"

        return unique_name

    def url(self, name):
        # Serve private images through the Django proxy.
        return reverse(
            "inspections:private_media",
            kwargs={"name": name},
        )
