"""
Jamo-level unigram BPE in icefall's format (<blk>=0, <sos/eos>=1, <unk>=2).

  python local/train_bpe_model.py --lang-dir $DATA/lang_bpe_500 --vocab-size 500 --transcript $DATA/lang_bpe_500/transcript.txt
"""
import argparse, shutil
from pathlib import Path
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-dir", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--vocab-size", type=int, default=500)
    ap.add_argument("--char-coverage", type=float, default=1.0, help="<1.0 maps rare syllables to <unk> (large corpora)")
    a = ap.parse_args()
    lang = Path(a.lang_dir); lang.mkdir(parents=True, exist_ok=True)
    prefix = str(lang / f"unigram_{a.vocab_size}")
    spm.SentencePieceTrainer.train(
        input=a.transcript, vocab_size=a.vocab_size, model_type="unigram", model_prefix=prefix,
        input_sentence_size=100_000_000, character_coverage=a.char_coverage,
        user_defined_symbols=["<blk>", "<sos/eos>"], unk_id=2, bos_id=-1, eos_id=-1,
        # jamo strings are long; allow pieces up to a whole short word
        max_sentencepiece_length=16, split_by_whitespace=True, remove_extra_whitespaces=True,
        # keep compatibility jamo as-is: the default nmt_nfkc rule would recompose them into syllables
        normalization_rule_name="identity",
    )
    shutil.copyfile(f"{prefix}.model", lang / "bpe.model")
    sp = spm.SentencePieceProcessor(model_file=str(lang / "bpe.model"))
    with open(lang / "tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
    print(f"vocab {sp.vocab_size()} → {lang/'bpe.model'}, {lang/'tokens.txt'}")


if __name__ == "__main__":
    main()
