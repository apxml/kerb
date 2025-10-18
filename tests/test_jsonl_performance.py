"""Tests for enhanced JSONL utilities with performance features."""

import json
import tempfile
from pathlib import Path

import pytest

from kerb.fine_tuning import jsonl


class TestCompression:
    """Test compression functionality."""

    def test_write_read_compressed(self):
        """Test writing and reading compressed files."""
        data = [{"id": i, "value": f"test_{i}"} for i in range(100)]

        with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as f:
            filepath = f.name

        try:
            # Write compressed
            jsonl.write_jsonl(data, filepath, compress=True)

            # Verify file is compressed (smaller than uncompressed)
            assert Path(filepath).exists()

            # Read compressed
            read_data = jsonl.read_jsonl(filepath)
            assert len(read_data) == len(data)
            assert read_data[0] == data[0]
            assert read_data[-1] == data[-1]
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_compression_types(self):
        """Test different compression types."""
        data = [{"test": "data"}]

        for comp_type, ext in [("gz", ".gz"), ("bz2", ".bz2"), ("xz", ".xz")]:
            with tempfile.NamedTemporaryFile(suffix=f".jsonl{ext}", delete=False) as f:
                filepath = f.name

            try:
                jsonl.write_jsonl(
                    data, filepath, compress=True, compression_type=comp_type
                )
                read_data = jsonl.read_jsonl(filepath)
                assert read_data == data
            finally:
                Path(filepath).unlink(missing_ok=True)


class TestStreaming:
    """Test streaming functionality."""

    def test_stream_jsonl_basic(self):
        """Test basic streaming."""
        data = [{"id": i} for i in range(1000)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        try:
            # Stream in batches
            batches = list(jsonl.stream_jsonl(filepath, batch_size=100))
            assert len(batches) == 10
            assert all(len(batch) == 100 for batch in batches)

            # Verify data
            all_items = [item for batch in batches for item in batch]
            assert len(all_items) == len(data)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_stream_with_filter(self):
        """Test streaming with filter function."""
        data = [{"id": i, "value": i % 2} for i in range(100)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        try:
            # Stream with filter (only even values)
            filtered = []
            for batch in jsonl.stream_jsonl(
                filepath, filter_fn=lambda x: x["value"] == 0
            ):
                filtered.extend(batch)

            assert len(filtered) == 50
            assert all(item["value"] == 0 for item in filtered)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_stream_with_transform(self):
        """Test streaming with transform function."""
        data = [{"value": i} for i in range(10)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        try:
            # Stream with transform
            transformed = []
            for batch in jsonl.stream_jsonl(
                filepath,
                transform_fn=lambda x: {
                    "value": x["value"],
                    "squared": x["value"] ** 2,
                },
            ):
                transformed.extend(batch)

            assert len(transformed) == 10
            assert transformed[3] == {"value": 3, "squared": 9}
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestParallelReading:
    """Test parallel reading functionality."""

    def test_parallel_read_jsonl(self):
        """Test parallel reading."""
        data = [{"id": i, "data": {"nested": i * 2}} for i in range(1000)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        try:
            # Read in parallel
            read_data = jsonl.parallel_read_jsonl(
                filepath, num_workers=2, chunk_size=100
            )
            assert len(read_data) == len(data)
            assert read_data[0] == data[0]
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestFiltering:
    """Test filtering functionality."""

    def test_filter_jsonl(self):
        """Test filtering JSONL file."""
        data = [{"id": i, "keep": i % 3 == 0} for i in range(30)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            input_file = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_file = f.name

        try:
            # Filter
            jsonl.filter_jsonl(input_file, output_file, filter_fn=lambda x: x["keep"])

            # Verify
            filtered_data = jsonl.read_jsonl(output_file)
            assert len(filtered_data) == 10
            assert all(item["keep"] for item in filtered_data)
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)


class TestTransformation:
    """Test transformation functionality."""

    def test_transform_jsonl(self):
        """Test transforming JSONL file."""
        data = [{"value": i} for i in range(10)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            input_file = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_file = f.name

        try:
            # Transform
            def transform(item):
                return {
                    "value": item["value"],
                    "doubled": item["value"] * 2,
                    "is_even": item["value"] % 2 == 0,
                }

            jsonl.transform_jsonl(input_file, output_file, transform)

            # Verify
            transformed_data = jsonl.read_jsonl(output_file)
            assert len(transformed_data) == 10
            assert transformed_data[3] == {"value": 3, "doubled": 6, "is_even": False}
            assert transformed_data[4] == {"value": 4, "doubled": 8, "is_even": True}
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)


class TestSplitting:
    """Test file splitting functionality."""

    def test_split_jsonl_by_line(self):
        """Test splitting JSONL file by line count."""
        data = [{"id": i} for i in range(25)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            input_file = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        try:
            # Split into chunks of 10
            with tempfile.TemporaryDirectory() as tmpdir:
                prefix = str(Path(tmpdir) / "chunk")
                output_files = jsonl.split_jsonl(
                    input_file, prefix, split_size=10, by_line=True
                )

                assert len(output_files) == 3  # 25 items -> 3 files (10, 10, 5)

                # Verify counts
                assert jsonl.count_jsonl_lines(output_files[0]) == 10
                assert jsonl.count_jsonl_lines(output_files[1]) == 10
                assert jsonl.count_jsonl_lines(output_files[2]) == 5
        finally:
            Path(input_file).unlink(missing_ok=True)


class TestDeduplication:
    """Test deduplication functionality."""

    def test_deduplicate_jsonl(self):
        """Test deduplicating JSONL file."""
        data = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
            {"id": 1, "text": "first"},  # duplicate
            {"id": 3, "text": "third"},
            {"id": 2, "text": "second"},  # duplicate
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            input_file = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_file = f.name

        try:
            # Deduplicate by id
            removed = jsonl.deduplicate_jsonl(
                input_file, output_file, key_fn=lambda x: x["id"], keep="first"
            )

            assert removed == 2

            # Verify
            deduped_data = jsonl.read_jsonl(output_file)
            assert len(deduped_data) == 3
            ids = [item["id"] for item in deduped_data]
            assert ids == [1, 2, 3]
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)


class TestSampling:
    """Test sampling functionality."""

    def test_sample_jsonl_by_count(self):
        """Test sampling by count."""
        data = [{"id": i} for i in range(100)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            input_file = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_file = f.name

        try:
            # Sample 10 items
            jsonl.sample_jsonl(input_file, output_file, n=10, random_state=42)

            sampled_data = jsonl.read_jsonl(output_file)
            assert len(sampled_data) == 10
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)

    def test_sample_jsonl_by_fraction(self):
        """Test sampling by fraction."""
        data = [{"id": i} for i in range(100)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            input_file = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_file = f.name

        try:
            # Sample 20%
            jsonl.sample_jsonl(input_file, output_file, fraction=0.2, random_state=42)

            sampled_data = jsonl.read_jsonl(output_file)
            assert len(sampled_data) == 20
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)


class TestMerging:
    """Test file merging functionality."""

    def test_merge_jsonl(self):
        """Test merging multiple JSONL files."""
        data1 = [{"source": 1, "id": i} for i in range(10)]
        data2 = [{"source": 2, "id": i} for i in range(10)]
        data3 = [{"source": 3, "id": i} for i in range(10)]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f1, tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f2, tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f3, tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False
        ) as fout:

            input_files = [f1.name, f2.name, f3.name]
            output_file = fout.name

            for item in data1:
                f1.write(json.dumps(item) + "\n")
            for item in data2:
                f2.write(json.dumps(item) + "\n")
            for item in data3:
                f3.write(json.dumps(item) + "\n")

        try:
            # Merge
            jsonl.merge_jsonl(input_files, output_file)

            # Verify
            merged_data = jsonl.read_jsonl(output_file)
            assert len(merged_data) == 30
        finally:
            for f in input_files + [output_file]:
                Path(f).unlink(missing_ok=True)


class TestStatistics:
    """Test statistics functionality."""

    def test_get_jsonl_stats(self):
        """Test getting JSONL file statistics."""
        data = [{"id": i, "text": f"Sample {i}", "value": i} for i in range(100)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        try:
            stats = jsonl.get_jsonl_stats(filepath, sample_size=50)

            assert stats["total_lines"] == 100
            assert stats["total_bytes"] > 0
            assert stats["compressed"] is False
            assert set(stats["keys"]) == {"id", "text", "value"}
            assert len(stats["sample_items"]) == 50
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestUtilities:
    """Test utility functions."""

    def test_count_jsonl_lines(self):
        """Test counting lines."""
        data = [{"id": i} for i in range(42)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name
            for item in data:
                f.write(json.dumps(item) + "\n")

        try:
            count = jsonl.count_jsonl_lines(filepath)
            assert count == 42
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_compress_decompress(self):
        """Test compression and decompression utilities."""
        data = [{"id": i, "data": f"test_{i}"} for i in range(10)]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            uncompressed_file = f.name

        try:
            # Write uncompressed
            jsonl.write_jsonl(data, uncompressed_file)

            # Compress
            compressed_file = jsonl.compress_jsonl(uncompressed_file)
            assert Path(compressed_file).exists()
            assert compressed_file.endswith(".gz")

            # Decompress
            decompressed_file = jsonl.decompress_jsonl(compressed_file)
            assert Path(decompressed_file).exists()

            # Verify data
            final_data = jsonl.read_jsonl(decompressed_file)
            assert final_data == data

            # Cleanup
            Path(compressed_file).unlink(missing_ok=True)
            Path(decompressed_file).unlink(missing_ok=True)
        finally:
            Path(uncompressed_file).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
