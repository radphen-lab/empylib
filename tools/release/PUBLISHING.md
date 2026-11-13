# Publish the two packages

GitHub environments named `pypi` are configured in both repositories.
The remaining account setup is PyPI Trusted Publishing. Use the exact values
below; the workflow filename has no `.github/workflows/` prefix.

| PyPI form field | Backend | empylib |
| --- | --- | --- |
| PyPI project name (pending publisher only) | `radphen-iadpython` | `empylib` |
| Owner | `PanxoPanza` | `radphen-lab` |
| Repository name | `iadpython` | `empylib` |
| Workflow name | `pypi.yaml` | `publish.yml` |
| Environment name | `pypi` | `pypi` |

For the new `radphen-iadpython` project, open
[account Publishing](https://pypi.org/manage/account/publishing/), select
GitHub, complete the pending publisher form, and click **Add**.
For existing `empylib`, open
[empylib Publishing](https://pypi.org/manage/project/empylib/settings/publishing/)
with a project-owner account, select GitHub, complete the matching publisher
form, and click **Add**. If a matching publisher already exists, retain it.
A pending publisher creates the new project only on its first successful
upload. [PyPI documents pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [publishers for existing projects](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

After the exact release commits and tags have passed the required CI:

1. In `PanxoPanza/iadpython`, run **Publish radphen-iadpython** (`pypi.yaml`)
   on tag `v0.7.0`, entering `0.7.0` for `confirm-version`. Approve the `pypi`
   environment if GitHub requests it. Wait for the upload and verify
   [radphen-iadpython 0.7.0](https://pypi.org/project/radphen-iadpython/0.7.0/).
2. In `radphen-lab/empylib`, run **Publish empylib 0.2.0** (`publish.yml`)
   on tag `v0.2.0`, entering `0.2.0`. Its gate downloads the backend from
   public PyPI and reruns the complete validation matrix. Approve the `pypi`
   environment if requested. Verify [empylib 0.2.0](https://pypi.org/project/empylib/0.2.0/).

Pushing commits or creating tags does not upload either package. The manual
publish workflows build, validate, and upload their own CI artifacts; local
`dist/` files are review evidence. Keep commits, tag targets, workflow run URLs,
and final statuses in the release handoff. Do not dispatch empylib publication
until the backend is publicly installable.

Install in a fresh virtual environment with `python -m pip install empylib==0.2.0`.
The backend distribution is `radphen-iadpython`, its Python import is
`iadpython`, and its CLI is `iadp`. Do not install upstream `iadpython` in the
same environment.
