import unittest

from pydantic import ValidationError

from research_system.schemas import EvidenceItemSchema, SpecialistFindingsSchema


class SchemaTests(unittest.TestCase):
    def test_evidence_item_accepts_valid_payload(self):
        item = EvidenceItemSchema(
            claim="Adoption is rising.",
            source_url="https://example.org/a",
            source_type="news_article",
            confidence=0.8,
            credibility=0.7,
            tags=["adoption"],
        )
        self.assertEqual(item.confidence, 0.8)

    def test_evidence_item_rejects_out_of_range_confidence(self):
        with self.assertRaises(ValidationError):
            EvidenceItemSchema(
                claim="x",
                source_url="https://example.org/a",
                source_type="news_article",
                confidence=1.5,
                credibility=0.7,
                tags=[],
            )

    def test_specialist_findings_wraps_evidence_list(self):
        findings = SpecialistFindingsSchema(
            evidence=[
                EvidenceItemSchema(
                    claim="x",
                    source_url="https://example.org/a",
                    source_type="news_article",
                    confidence=0.5,
                    credibility=0.5,
                    tags=[],
                )
            ],
            summary="Short summary.",
        )
        self.assertEqual(len(findings.evidence), 1)


if __name__ == "__main__":
    unittest.main()
