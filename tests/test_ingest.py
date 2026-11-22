import datetime as dt
import logging
from pathlib import Path

import pytest

from review_radar.ingest import (
    MAX_TEXT_LENGTH,
    IngestError,
    IngestResult,
    RowError,
    RowErrorReason,
    read_reviews,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO_ROOT / "data" / "sample" / "reviews.csv"

HEADER = b"review_id,review_date,rating,text,source\n"
FAKE_TEXT = "ZEBRA-QUOKKA-9917 distinctive fake review"


def write(tmp_path: Path, content: bytes, name: str = "reviews.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def reasons(result: IngestResult) -> list[tuple[int, RowErrorReason]]:
    return [(e.line, e.reason) for e in result.errors]


class TestHappyPath:
    def test_reads_all_fields(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great stuff,web\n")

        result = read_reviews(path)

        assert result.errors == ()
        assert result.rows_read == 1
        (review,) = result.reviews
        assert review.external_id == "r1"
        assert review.review_date == dt.date(2026, 11, 10)
        assert review.rating == 5
        assert review.text == "Great stuff"
        assert review.source == "web"

    def test_optional_columns_may_be_absent(self, tmp_path: Path) -> None:
        path = write(tmp_path, b"review_date,text\n2026-11-10,Fine\n")

        (review,) = read_reviews(path).reviews

        assert (review.external_id, review.rating, review.source) == (None, None, None)

    def test_blank_optional_values_become_none(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + b" ,2026-11-10, ,Fine, \n")

        (review,) = read_reviews(path).reviews

        assert (review.external_id, review.rating, review.source) == (None, None, None)

    def test_text_is_stripped(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + b'r1,2026-11-10,3,"  padded  ",web\n')

        assert read_reviews(path).reviews[0].text == "padded"

    def test_no_trailing_newline(self, tmp_path: Path) -> None:
        result = read_reviews(write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web"))

        assert len(result.reviews) == 1


class TestFormats:
    def test_utf8_bom(self, tmp_path: Path) -> None:
        path = write(tmp_path, b"\xef\xbb\xbf" + HEADER + b"r1,2026-11-10,5,Great,web\n")

        result = read_reviews(path)

        assert len(result.reviews) == 1
        assert result.errors == ()

    def test_semicolon_delimiter_autodetected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            b"review_id;review_date;rating;text;source\nr1;2026-11-10;4;Nice, really;web\n",
        )

        (review,) = read_reviews(path).reviews

        assert review.text == "Nice, really"
        assert review.rating == 4

    def test_comma_delimiter_autodetected_with_semicolon_in_text(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + b'r1,2026-11-10,4,"Nice; really",web\n')

        assert read_reviews(path).reviews[0].text == "Nice; really"

    def test_delimiter_override(self, tmp_path: Path) -> None:
        path = write(tmp_path, b"review_date|text\n2026-11-10|Fine\n")

        with pytest.raises(IngestError):
            read_reviews(path)
        assert read_reviews(path, delimiter="|").reviews[0].text == "Fine"

    @pytest.mark.parametrize("delimiter", ["", ";;", '"', "x", " ", "\n", ":"])
    def test_invalid_delimiter_rejected(self, tmp_path: Path, delimiter: str) -> None:
        path = write(tmp_path, HEADER)

        with pytest.raises(IngestError, match="delimiter"):
            read_reviews(path, delimiter=delimiter)

    @pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
    def test_allowed_delimiters(self, tmp_path: Path, delimiter: str) -> None:
        d = delimiter.encode()
        content = b"review_date" + d + b"text\n2026-11-10" + d + b"Fine\n"

        result = read_reviews(write(tmp_path, content), delimiter=delimiter)

        assert result.reviews[0].text == "Fine"

    def test_quoted_comma_and_quote(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + b'r1,2026-11-10,5,"Good, very ""good""",web\n')

        assert read_reviews(path).reviews[0].text == 'Good, very "good"'

    def test_quoted_newline_keeps_later_line_numbers_correct(self, tmp_path: Path) -> None:
        content = (
            HEADER  # line 1
            + b'r1,2026-11-10,5,"first line\nsecond line\nthird line",web\n'  # lines 2-4
            + b"r2,not-a-date,5,Fine,web\n"  # line 5
        )

        result = read_reviews(write(tmp_path, content))

        assert result.reviews[0].text == "first line\nsecond line\nthird line"
        assert reasons(result) == [(5, RowErrorReason.INVALID_DATE)]

    def test_error_in_multiline_record_reports_start_line(self, tmp_path: Path) -> None:
        content = HEADER + b'r1,nope,5,"two\nlines",web\n'

        result = read_reviews(write(tmp_path, content))

        assert reasons(result) == [(2, RowErrorReason.INVALID_DATE)]

    def test_crlf_line_endings(self, tmp_path: Path) -> None:
        content = (
            b"review_id,review_date,rating,text,source\r\n"
            b"r1,2026-11-10,5,Great,web\r\n"
            b"r2,bad,5,Fine,web\r\n"
            b"r3,2026-11-11,2,Poor,app\r\n"
        )

        result = read_reviews(write(tmp_path, content))

        assert [r.external_id for r in result.reviews] == ["r1", "r3"]
        assert reasons(result) == [(3, RowErrorReason.INVALID_DATE)]

    def test_non_ascii_round_trip(self, tmp_path: Path) -> None:
        text = "Отличный товар, café \U0001f60d naïve"
        content = HEADER + f'r1,2026-11-10,5,"{text}",web\n'.encode()

        assert read_reviews(write(tmp_path, content)).reviews[0].text == text

    def test_physically_empty_lines_are_ignored_and_not_counted(self, tmp_path: Path) -> None:
        content = b"\n" + HEADER + b"\nr1,2026-11-10,5,Great,web\n\nr2,bad,5,Fine,web\n\n"

        result = read_reviews(write(tmp_path, content))

        assert result.rows_read == 2
        assert len(result.reviews) == 1
        assert reasons(result) == [(6, RowErrorReason.INVALID_DATE)]

    def test_whitespace_only_leading_lines_before_header_are_ignored(self, tmp_path: Path) -> None:
        content = b"  \n\n" + HEADER + b"r1,2026-11-10,5,Great,web\n"

        assert len(read_reviews(write(tmp_path, content)).reviews) == 1

    @pytest.mark.parametrize("line", [b'""', b"   ", b"\t"])
    def test_non_empty_blank_looking_rows_are_counted(self, tmp_path: Path, line: bytes) -> None:
        content = HEADER + b"r1,2026-11-10,5,Great,web\n" + line + b"\nr2,bad,5,Fine,web\n"

        result = read_reviews(write(tmp_path, content))

        assert result.rows_read == 3
        assert result.rows_read == len(result.reviews) + len(result.errors)
        assert reasons(result) == [
            (3, RowErrorReason.WRONG_FIELD_COUNT),
            (4, RowErrorReason.INVALID_DATE),
        ]

    @pytest.mark.parametrize(
        "header",
        [
            b"REVIEW_DATE,TEXT",
            b"  Review_Date , Text  ",
            b'"review_date","text"',
        ],
    )
    def test_header_case_and_whitespace(self, tmp_path: Path, header: bytes) -> None:
        path = write(tmp_path, header + b"\n2026-11-10,Fine\n")

        assert read_reviews(path).reviews[0].text == "Fine"

    def test_extra_unknown_columns_are_ignored_and_reported(self, tmp_path: Path) -> None:
        path = write(tmp_path, b"review_date,text,agent, Ratting \n2026-11-10,Fine,Sam,5\n")

        result = read_reviews(path)

        assert result.reviews[0].text == "Fine"
        assert result.ignored_columns == ("agent", "Ratting")

    def test_no_ignored_columns_by_default(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Fine,web\n")

        assert read_reviews(path).ignored_columns == ()

    def test_ignored_column_names_are_truncated(self, tmp_path: Path) -> None:
        path = write(tmp_path, b"review_date,text," + b"z" * 100 + b"\n2026-11-10,Fine,x\n")

        assert read_reviews(path).ignored_columns == ("z" * 40,)

    def test_blank_header_names_are_reported_as_blank(self, tmp_path: Path) -> None:
        path = write(tmp_path, b"review_date,text,, \n2026-11-10,Fine,x,y\n")

        result = read_reviews(path)

        assert result.ignored_columns == ("(blank)", "(blank)")
        assert len(result.reviews) == 1


class TestRowErrors:
    def test_missing_text(self, tmp_path: Path) -> None:
        content = HEADER + b"r1,2026-11-10,5,,web\nr2,2026-11-10,5,   ,web\n"

        result = read_reviews(write(tmp_path, content))

        assert reasons(result) == [
            (2, RowErrorReason.MISSING_TEXT),
            (3, RowErrorReason.MISSING_TEXT),
        ]

    def test_text_length_boundary(self, tmp_path: Path) -> None:
        ok = "a" * MAX_TEXT_LENGTH
        padded_ok = "  " + ok + "  "
        too_long = "a" * (MAX_TEXT_LENGTH + 1)
        content = (
            HEADER
            + (
                f"r1,2026-11-10,5,{ok},web\nr2,2026-11-10,5,{padded_ok},web\n"
                f"r3,2026-11-10,5,{too_long},web\n"
            ).encode()
        )

        result = read_reviews(write(tmp_path, content))

        assert [r.external_id for r in result.reviews] == ["r1", "r2"]
        assert reasons(result) == [(4, RowErrorReason.TEXT_TOO_LONG)]

    @pytest.mark.parametrize(
        "value",
        ["", "10/11/2026", "2026-13-01", "2026-02-30", "20261110", "2026-W46-1", "yesterday"],
    )
    def test_invalid_date(self, tmp_path: Path, value: str) -> None:
        path = write(tmp_path, HEADER + f"r1,{value},5,Fine,web\n".encode())

        assert reasons(read_reviews(path)) == [(2, RowErrorReason.INVALID_DATE)]

    @pytest.mark.parametrize("value", ["4.0", "five", "0", "6", "-1", "+4", "04", "1,5", "٤"])
    def test_invalid_rating(self, tmp_path: Path, value: str) -> None:
        path = write(tmp_path, HEADER + f'r1,2026-11-10,"{value}",Fine,web\n'.encode())

        assert reasons(read_reviews(path)) == [(2, RowErrorReason.INVALID_RATING)]

    @pytest.mark.parametrize("value", ["1", "2", "3", "4", "5", " 3 ", ""])
    def test_valid_rating(self, tmp_path: Path, value: str) -> None:
        path = write(tmp_path, HEADER + f"r1,2026-11-10,{value},Fine,web\n".encode())

        result = read_reviews(path)

        assert result.errors == ()
        assert result.reviews[0].rating == (int(value) if value.strip() else None)

    def test_too_many_fields(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Fine,web,extra\n")

        assert reasons(read_reviews(path)) == [(2, RowErrorReason.WRONG_FIELD_COUNT)]

    def test_too_few_fields(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + b"r1,2026-11-10,5\nr2,2026-11-10,5,Fine,web\n")

        result = read_reviews(path)

        assert reasons(result) == [(2, RowErrorReason.WRONG_FIELD_COUNT)]
        assert len(result.reviews) == 1

    def test_malformed_row_is_skipped_and_reading_continues(self, tmp_path: Path) -> None:
        content = (
            HEADER
            + b"r1,2026-11-10,5,Great,web\n"  # line 2
            + b'r2,2026-11-10,"abc"x,web\n'  # line 3: malformed
            + b"r3,2026-11-10,5,Fine,web\n"  # line 4
            + b"r4,bad,5,Fine,web\n"  # line 5
        )

        result = read_reviews(write(tmp_path, content))

        assert [r.external_id for r in result.reviews] == ["r1", "r3"]
        assert reasons(result) == [
            (3, RowErrorReason.MALFORMED_ROW),
            (5, RowErrorReason.INVALID_DATE),
        ]
        assert result.rows_read == 4

    def test_malformed_row_after_multiline_record_keeps_line_numbers(self, tmp_path: Path) -> None:
        content = (
            HEADER
            + b'r1,2026-11-10,5,"two\nlines",web\n'  # lines 2-3
            + b'r2,2026-11-10,"abc"x,web\n'  # line 4
            + b"r3,bad,5,Fine,web\n"  # line 5
        )

        result = read_reviews(write(tmp_path, content))

        assert reasons(result) == [
            (4, RowErrorReason.MALFORMED_ROW),
            (5, RowErrorReason.INVALID_DATE),
        ]

    def test_malformed_row_ending_a_multiline_record_reports_start_line(
        self, tmp_path: Path
    ) -> None:
        content = (
            HEADER
            + b'r1,2026-11-10,5,"first\nsecond"x,web\n'  # lines 2-3, error on line 3
            + b"r2,bad,5,Fine,web\n"  # line 4
        )

        # The closing quote is on the error line, so the line has an odd quote count and
        # resuming is not provably safe: this is a file-level error.
        with pytest.raises(IngestError, match="line 2"):
            read_reviews(write(tmp_path, content))

    def test_unquoted_comma_in_text_is_wrong_field_count(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Good, honestly,web\n")

        assert reasons(read_reviews(path)) == [(2, RowErrorReason.WRONG_FIELD_COUNT)]

    def test_rows_read_equals_reviews_plus_errors(self, tmp_path: Path) -> None:
        content = (
            HEADER
            + b"r1,2026-11-10,5,Great,web\n"
            + b"r2,bad,5,Fine,web\n"
            + b"r3,2026-11-10,9,Fine,web\n"
            + b"r4,2026-11-10,5,,web\n"
            + b"short\n"
        )

        result = read_reviews(write(tmp_path, content))

        assert result.rows_read == 5
        assert result.rows_read == len(result.reviews) + len(result.errors)

    def test_skipped_by_reason(self, tmp_path: Path) -> None:
        content = (
            HEADER + b"r1,bad,5,Fine,web\n" + b"r2,2026-11-10,5,,web\n" + b"r3,worse,5,Fine,web\n"
        )

        counts = read_reviews(write(tmp_path, content)).skipped_by_reason()

        assert counts == {RowErrorReason.INVALID_DATE: 2, RowErrorReason.MISSING_TEXT: 1}

    def test_skipped_by_reason_empty(self) -> None:
        assert IngestResult(reviews=(), errors=(), rows_read=0).skipped_by_reason() == {}

    def test_row_error_is_frozen_value(self) -> None:
        error = RowError(line=3, reason=RowErrorReason.INVALID_DATE)

        assert error == RowError(3, RowErrorReason.INVALID_DATE)
        assert error.reason == "invalid_date"
        with pytest.raises(AttributeError):
            error.line = 4  # type: ignore[misc]


class TestFileErrors:
    def test_duplicate_review_id_keeps_first_and_flags_later(self, tmp_path: Path) -> None:
        content = (
            HEADER
            + b"r1,2026-11-10,5,First,web\n"
            + b"r2,2026-11-10,5,Other,web\n"
            + b"r1,2026-11-11,1,Second,app\n"
            + b" r1 ,2026-11-11,1,Third,app\n"
        )

        result = read_reviews(write(tmp_path, content))

        assert [r.text for r in result.reviews] == ["First", "Other"]
        assert reasons(result) == [
            (4, RowErrorReason.DUPLICATE_REVIEW_ID),
            (5, RowErrorReason.DUPLICATE_REVIEW_ID),
        ]
        assert result.rows_read == 4

    def test_blank_review_ids_are_never_duplicates(self, tmp_path: Path) -> None:
        content = HEADER + b",2026-11-10,5,A,web\n ,2026-11-10,5,B,web\n,2026-11-10,5,C,web\n"

        result = read_reviews(write(tmp_path, content))

        assert len(result.reviews) == 3
        assert result.errors == ()

    def test_invalid_first_occurrence_does_not_reserve_the_id(self, tmp_path: Path) -> None:
        content = HEADER + b"r1,bad,5,A,web\nr1,2026-11-10,5,B,web\n"

        result = read_reviews(write(tmp_path, content))

        assert [r.text for r in result.reviews] == ["B"]
        assert reasons(result) == [(2, RowErrorReason.INVALID_DATE)]

    def test_missing_required_column(self, tmp_path: Path) -> None:
        path = write(tmp_path, b"review_id,rating,text\nr1,5,Fine\n")

        with pytest.raises(IngestError, match="review_date"):
            read_reviews(path)

    def test_missing_text_column(self, tmp_path: Path) -> None:
        path = write(tmp_path, b"review_date,rating\n2026-11-10,5\n")

        with pytest.raises(IngestError, match="text"):
            read_reviews(path)

    def test_duplicate_column(self, tmp_path: Path) -> None:
        path = write(tmp_path, b"review_date,text,Text\n2026-11-10,a,b\n")

        with pytest.raises(IngestError, match="[Dd]uplicate"):
            read_reviews(path)

    @pytest.mark.parametrize("content", [b"", b"\n\n  \n", b"\xef\xbb\xbf"])
    def test_empty_file(self, tmp_path: Path, content: bytes) -> None:
        with pytest.raises(IngestError, match="empty"):
            read_reviews(write(tmp_path, content))

    def test_header_only_file_has_no_data_rows(self, tmp_path: Path) -> None:
        with pytest.raises(IngestError, match="The file has no data rows."):
            read_reviews(write(tmp_path, HEADER))

    def test_header_and_empty_lines_only_has_no_data_rows(self, tmp_path: Path) -> None:
        with pytest.raises(IngestError, match="no data rows"):
            read_reviews(write(tmp_path, HEADER + b"\n\n"))

    def test_bare_cr_is_rejected_with_first_line_number(self, tmp_path: Path) -> None:
        content = HEADER + b"r1,2026-11-10,5,Fine,web\r\nr2,2026-11-10,5,Old\rMac,web\n"

        with pytest.raises(IngestError, match=r"bare CR line breaks.*LF or CRLF") as excinfo:
            read_reviews(write(tmp_path, content))
        assert "line 3" in str(excinfo.value)

    def test_bare_cr_inside_quoted_field_is_rejected(self, tmp_path: Path) -> None:
        content = HEADER + b'r1,2026-11-10,5,"a\rb",web\n'

        with pytest.raises(IngestError, match="bare CR"):
            read_reviews(write(tmp_path, content))

    def test_bare_cr_at_end_of_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(IngestError, match="bare CR"):
            read_reviews(write(tmp_path, HEADER + b"r1,2026-11-10,5,Fine,web\r"))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(IngestError, match="read"):
            read_reviews(tmp_path / "nope.csv")

    def test_directory_instead_of_file(self, tmp_path: Path) -> None:
        with pytest.raises(IngestError, match="read"):
            read_reviews(tmp_path)

    def test_cp1251_bytes_with_default_encoding(self, tmp_path: Path) -> None:
        content = HEADER + "r1,2026-11-10,5,Отличный товар,web\n".encode("cp1251")

        with pytest.raises(IngestError, match="(?i)encoding"):
            read_reviews(write(tmp_path, content))

    def test_cp1251_with_explicit_encoding(self, tmp_path: Path) -> None:
        content = HEADER + "r1,2026-11-10,5,Отличный товар,web\n".encode("cp1251")

        result = read_reviews(write(tmp_path, content), encoding="cp1251")

        assert result.reviews[0].text == "Отличный товар"

    def test_unknown_encoding(self, tmp_path: Path) -> None:
        with pytest.raises(IngestError, match="(?i)encoding"):
            read_reviews(write(tmp_path, HEADER), encoding="no-such-codec")

    def test_nul_byte(self, tmp_path: Path) -> None:
        content = HEADER + b"r1,2026-11-10,5,Fine,web\nr2,2026-11-10,5,Bro\x00ken,web\n"

        with pytest.raises(IngestError, match=r"(?i)NUL.*line 3"):
            read_reviews(write(tmp_path, content))

    def test_field_larger_than_csv_limit(self, tmp_path: Path) -> None:
        huge = b"a" * 200_000
        content = HEADER + b"r1,2026-11-10,5,Fine,web\nr2,2026-11-10,5," + huge + b",web\n"

        with pytest.raises(IngestError, match="line 3"):
            read_reviews(write(tmp_path, content))

    def test_unterminated_quote_is_an_error_not_a_swallowed_file(self, tmp_path: Path) -> None:
        content = HEADER + b'r1,2026-11-10,5,"never closed,web\nr2,2026-11-10,5,Fine,web\n'

        with pytest.raises(IngestError, match="line 2"):
            read_reviews(write(tmp_path, content))

    def test_malformed_row_whose_tail_may_open_a_multiline_field_is_an_error(
        self, tmp_path: Path
    ) -> None:
        # After the error csv cannot tell where the record really ends; resuming would
        # parse the continuation line as a bogus record, so this stays file-level.
        content = HEADER + b'r1,2026-11-10,"a"z,"q\nw",web\nr2,2026-11-10,5,Fine,web\n'

        with pytest.raises(IngestError, match="line 2"):
            read_reviews(write(tmp_path, content))


class TestPrivacy:
    def test_review_text_never_in_logs_or_row_errors(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        long_text = FAKE_TEXT + "x" * MAX_TEXT_LENGTH
        content = (
            HEADER
            + f"r1,not-a-date,5,{FAKE_TEXT},web\n".encode()
            + f"r2,2026-11-10,7,{FAKE_TEXT},web\n".encode()
            + f"r3,2026-11-10,5,{long_text},web\n".encode()
            + f"r4,2026-11-10,5,{FAKE_TEXT},web,extra\n".encode()
            + f"r5,2026-11-10,5,{FAKE_TEXT},web\n".encode()
            + f'r6,2026-11-10,5,"{FAKE_TEXT}"x,web\n'.encode()  # malformed_row
            + f"r5,2026-11-10,5,{FAKE_TEXT},web\n".encode()  # duplicate_review_id
        )
        path = write(tmp_path, content)

        with caplog.at_level(logging.DEBUG):
            result = read_reviews(path)

        assert len(result.errors) == 6
        assert {e.reason for e in result.errors} >= {
            RowErrorReason.MALFORMED_ROW,
            RowErrorReason.DUPLICATE_REVIEW_ID,
        }
        assert FAKE_TEXT not in caplog.text
        assert FAKE_TEXT not in repr((result.errors, result.ignored_columns))

    def test_headerless_file_never_reports_its_text_as_ignored_columns(
        self, tmp_path: Path
    ) -> None:
        content = (FAKE_TEXT + ",2026-11-10\n" + FAKE_TEXT + ",x\n").encode()

        with pytest.raises(IngestError) as excinfo:
            read_reviews(write(tmp_path, content))

        assert FAKE_TEXT not in str(excinfo.value)

    def test_ignored_columns_hold_only_header_names(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        content = f"review_date,text,agent\n2026-11-10,{FAKE_TEXT},Sam\n".encode()

        with caplog.at_level(logging.DEBUG):
            result = read_reviews(write(tmp_path, content))

        assert result.ignored_columns == ("agent",)
        assert FAKE_TEXT not in caplog.text
        assert FAKE_TEXT not in repr(result.ignored_columns)

    @pytest.mark.parametrize(
        "content",
        [
            (FAKE_TEXT + ",2026-11-10\n" + FAKE_TEXT + "\n").encode(),  # no real header
            ("review_date,rating\n2026-11-10," + FAKE_TEXT + "\n").encode(),
            ('review_date,text\n2026-11-10,"' + FAKE_TEXT + "\n").encode(),  # unterminated
            ("review_date,text\n2026-11-10," + FAKE_TEXT + "\x00\n").encode(),  # NUL
            ("review_date,text\n2026-11-10," + FAKE_TEXT + "\r2026\n").encode(),  # bare CR
            b"review_date,text\n",  # no data rows
            ('review_date,text\n2026-11-10,"' + FAKE_TEXT + '"x,"q\nw"\n').encode(),  # ambiguous
        ],
    )
    def test_ingest_error_has_no_review_text(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, content: bytes
    ) -> None:
        with caplog.at_level(logging.DEBUG), pytest.raises(IngestError) as excinfo:
            read_reviews(write(tmp_path, content))

        assert FAKE_TEXT not in str(excinfo.value)
        assert FAKE_TEXT not in caplog.text

    def test_logs_one_info_summary_with_counts(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        content = HEADER + b"r1,2026-11-10,5,Great,web\nr2,bad,5,Fine,web\n"

        with caplog.at_level(logging.INFO, logger="review_radar.ingest"):
            read_reviews(write(tmp_path, content))

        records = [r for r in caplog.records if r.name == "review_radar.ingest"]
        assert len(records) == 1
        assert records[0].levelno == logging.INFO
        assert records[0].getMessage() == "Ingest finished: 2 rows read, 1 imported, 1 skipped"


class TestCommittedSample:
    def test_sample_reads_cleanly(self) -> None:
        result = read_reviews(SAMPLE_CSV)

        assert len(result.reviews) == 300
        assert result.errors == ()
        assert result.rows_read == 300
