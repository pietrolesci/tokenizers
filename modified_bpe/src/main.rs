use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence, NormalizerWrapper, byte_level};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, NormalizedString, Result, Tokenizer, TokenizerBuilder, TokenizerImpl};
// use tokenizers::{Tokenizer};


fn main() -> Result<()> {
    
    // let mut old_tok = Tokenizer::from_pretrained("gpt2", None)?;
    // println!("Normalizer: {:?}", old_tok.get_normalizer());
    // println!("Pretok: {:?}", old_tok.get_pre_tokenizer());
    // println!("Postproc: {:?}", old_tok.get_post_processor());
    // println!("Model: {:?}", old_tok.get_model());
    // println!("Decode: {:?}", old_tok.get_decoder());
    // println!("Vocab size: {:?}", old_tok.get_vocab_size(true));

    // define the trainer
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(2000)
        .min_frequency(2)
        .special_tokens(vec![
            AddedToken::from(String::from("|endoftext|"), true),
        ])
        .build();

    // define tokenizer
    let mut new_tok: TokenizerImpl<BPE, NormalizerWrapper, ByteLevel, ByteLevel, ByteLevel> = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_pre_tokenizer(Some(ByteLevel::new(false, true, true)))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    println!("Normalizer: {:?}", new_tok.get_normalizer());
    println!("Pretok: {:?}", new_tok.get_pre_tokenizer());
    println!("Postproc: {:?}", new_tok.get_post_processor());
    println!("Model: {:?}", new_tok.get_model());
    println!("Decode: {:?}", new_tok.get_decoder());
    println!("Vocab size: {:?}", new_tok.get_vocab_size(true));
    println!("Special tokens: {:?}", new_tok.get_encode_special_tokens());

    // train
    new_tok.train_from_files(&mut trainer, vec!["data/test_small.txt".to_string()])?;

    
    let output = new_tok.encode("Hello, y'all! How are you 😁 ?", true)?;
    println!("Hello, y'all! How are you 😁 ?");
    println!("{:?}", output.get_ids());
    println!("{:?}", output.get_tokens());
    let decoded = new_tok.decode(output.get_ids(), false)?;
    println!("{:?}", decoded);

    new_tok.save("tokenizer.json", true)?;
    
    
    let mut tokenizer = Tokenizer::from_file("tokenizer.json")?;
    println!("{:?}", tokenizer);
    let output = tokenizer.encode("Hello, y'all! How are you 😁 ?", true)?;
    println!("Hello, y'all! How are you 😁 ?");
    println!("{:?}", output.get_ids());
    println!("{:?}", output.get_tokens());
    let decoded = tokenizer.decode(output.get_ids(), false)?;
    println!("{:?}", decoded);


    // save


    // println!("Final vocab: {:?}", tokenizer.get_vocab(true));

    Ok(())
}
