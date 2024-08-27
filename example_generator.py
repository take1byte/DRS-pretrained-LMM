# a case of drs-text generation
from tokenization_mlm import MLMTokenizer
from transformers import MBartForConditionalGeneration

class ExampleGenerator():
    def __init__(self):
        # For DRS parsing, src_lang should be set to en_XX, de_DE, it_IT, or nl_XX
        self.gen_tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='<drs>')
        self.parse_tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='en_XX')
        self.model = MBartForConditionalGeneration.from_pretrained('laihuiyuan/DRS-LMM')

    def drs2en(self, gold_drs: str) -> str:
        inp_ids = self.gen_tokenizer.encode(gold_drs, return_tensors="pt")
        forced_ids = self.gen_tokenizer.encode("en_XX", add_special_tokens=False, return_tensors="pt")
        outs = self.model.generate(input_ids=inp_ids, forced_bos_token_id=forced_ids.item(), num_beams=5, max_length=150)
        text = self.gen_tokenizer.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text

    def mkGenExample(self, text: str, drs: str) -> None:
        predicted_text = self.drs2en(drs)
        print(f"GOLD TEXT: {text}\nPREDICTED TEXT: {predicted_text}\nDRS: {drs}\n")

    def en2drs(self, text: str) -> str:
        inp_ids = self.parse_tokenizer.encode(text, return_tensors='pt')
        forced_ids = self.parse_tokenizer.encode('<drs>', add_special_tokens=False, return_tensors="pt")
        outs = self.model.generate(input_ids=inp_ids, forced_bos_token_id=forced_ids.item(), num_beams=5, max_length=150)
        drs = self.parse_tokenizer.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return drs

    def mkParseExample(self, text: str, gold_drs: str) -> None:
        predicted_drs = self.en2drs(text)
        predicted_text = self.drs2en(predicted_drs)
        print(f"TEXT: {text}\nGOLD DRS: {gold_drs}\nPREDICTED DRS: {predicted_drs}\nPREDICTED TEXT: {predicted_text}\n")
