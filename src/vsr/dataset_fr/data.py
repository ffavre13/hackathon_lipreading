import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    import polars as pl

    repo_id = "sindhuhegde/multivsr"
    out = Path(__file__).resolve().parent / "data_csv"
    out.mkdir(exist_ok=True)

    splits = [
        "train_major",
        "train_minor",
        "val_major",
        "val_minor",
        "test",
    ]

    all_ids = set()

    # download the csvs and filter for french segments
    for split in splits:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{split}.csv",
            repo_type="dataset",
        )

        df = pl.read_csv(path)
        df_fr = df.filter(pl.col("language") == "fr")

        df_fr.write_csv(out / f"{split}_fr.csv")

        ids = df_fr["file_path"].str.split("/").list.first()
        all_ids.update(ids.drop_nulls().to_list())

        print(split, len(df_fr), "french segs")

    with open(out / "ytids_fr.txt", "w") as f:
        f.write("\n".join(sorted(all_ids)) + "\n")

    print("Distinct youtube vids :", len(all_ids))
    return out, pl


@app.cell
def _(out, pl):
    from download_videos import get_cuts_from_pckl

    video_ids = pl.read_csv(out / "ytids_fr.txt", has_header=False).to_series().to_list()
    get_cuts_from_pckl(video_ids[1], out.parent / "pckl_fr")
    return


if __name__ == "__main__":
    app.run()
