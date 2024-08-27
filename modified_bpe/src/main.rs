use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence, NormalizerWrapper};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Result, TokenizerBuilder};

fn main() -> Result<()> {
    let vocab_size: usize = 2000;

    // define the trainer
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(0)
        .special_tokens(vec![
            AddedToken::from(String::from("|endoftext|"), true),
        ])
        .build();

    // define tokenizer
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            // Strip::new(true, true).into(),
            // NFC.into(),
        ])))
        // .with_normalizer()
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    // train
    tokenizer.train_from_files(&mut trainer, vec!["data/test.txt".to_string()])?;

    // save
    tokenizer.save("tokenizer.json", true)?;

    println!("Final vocab: {:?}", tokenizer.get_vocab(true));

    Ok(())
}
