import os
from pathlib import Path

from huggingface_hub import HfApi


def _require_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN before uploading to Hugging Face.")
    return token


class HuggingFaceShardUploader:
    """Uploads completed dataset artifacts and optionally removes local copies."""

    def __init__(self, repo_id, repo_type="dataset", token=None):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = token or _require_hf_token()
        self.api = HfApi(token=self.token)

    def ensure_repo(self, private=True):
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            private=private,
            exist_ok=True,
        )

    def upload_file(self, local_path, repo_path, delete_local=False, commit_message=None):
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(local_path)

        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message=commit_message or f"Upload {repo_path}",
        )

        if delete_local:
            local_path.unlink()

    def upload_directory_files(self, local_dir, repo_prefix="", delete_local=False, private=True):
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise NotADirectoryError(local_dir)

        self.ensure_repo(private=private)
        uploaded = []
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            repo_path = str(Path(repo_prefix) / path.relative_to(local_dir)) if repo_prefix else str(path.relative_to(local_dir))
            self.upload_file(
                path,
                repo_path,
                delete_local=delete_local,
                commit_message=f"Upload dataset artifact {repo_path}",
            )
            uploaded.append(repo_path)
        return uploaded
