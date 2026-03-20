from app.services.search_context_service import CondensedSearchContext, SearchContextService


def test_prepare_documents_trims_content():
    service = SearchContextService()
    service.max_docs = 1
    service.max_chars_per_doc = 20

    prepared = service._prepare_documents(
        [
            {
                "c_id": "doc-1",
                "creation_year": 2024,
                "domain": "medical",
                "source": "who",
                "source_spec": "guideline",
                "content": "첫 번째 문서 내용이 아주 길어서 잘려야 합니다. 여러 공백도   정리됩니다.",
            },
            {
                "c_id": "doc-2",
                "content": "두 번째 문서",
            },
        ]
    )

    assert len(prepared) == 1
    assert prepared[0]["content"].endswith("...")
    assert "  " not in prepared[0]["content"]


def test_fallback_context_contains_compact_evidence():
    service = SearchContextService()
    context = service._build_fallback_context(
        question="잠복결핵이 뭐야?",
        documents=[
            {
                "c_id": "25276_3",
                "creation_year": 2000,
                "source_spec": "who",
                "content": "잠복결핵감염은 증상이 없지만 향후 활동성 결핵으로 진행할 수 있다.",
            }
        ],
        context_type="medical",
    )

    assert "[의학 검색 요약]" in context
    assert "잠복결핵이 뭐야?" in context
    assert "25276_3" in context
    assert "excerpt=" in context


def test_format_condensed_context_is_short_and_structured():
    service = SearchContextService()
    context = service._format_condensed_context(
        question="기침이 3주째인데 결핵일 수 있어?",
        documents=[
            {
                "c_id": "doc-1",
                "creation_year": 2022,
                "source_spec": "cdc",
                "content": "unused",
            }
        ],
        context_type="medical",
        condensed=CondensedSearchContext(
            summary="2주 이상 기침은 결핵 평가 필요성과 관련될 수 있다.",
            key_points=[
                "2주 이상 지속되는 기침은 결핵 의심 신호로 자주 언급된다.",
                "확진에는 검사와 의료진 판단이 필요하다.",
            ],
            cautions=["검색 문서만으로 개인 진단은 불가능하다."],
        ),
    )

    assert "요약:" in context
    assert "핵심 포인트:" in context
    assert "주의사항:" in context
    assert "참고 문서:" in context
