from dify_plugin.interfaces.model.openai_compatible.llm import OAICompatLargeLanguageModel

from models.llm.llm import FlyfusLargeLanguageModel


def test_long_text_is_counted_in_local_tokenizer_chunks(monkeypatch) -> None:
    chunk_lengths: list[int] = []

    def count_chunk(_self, text: str) -> int:
        chunk_lengths.append(len(text))
        return len(text) // 4

    monkeypatch.setattr(
        OAICompatLargeLanguageModel,
        "_get_num_tokens_by_gpt2",
        count_chunk,
    )
    model = FlyfusLargeLanguageModel({})

    token_count = model._get_num_tokens_by_gpt2("x" * 210_000)

    assert chunk_lengths == [80_000, 80_000, 50_000]
    assert token_count == 52_500
