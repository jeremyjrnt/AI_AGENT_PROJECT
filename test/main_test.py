#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for advanced/full pipeline tests.
Moved from project root to test/ folder.
"""

from test_rag_pipeline import test_rag_pipeline


def main():
    # You can add more advanced or integration tests here
    test_rag_pipeline()


if __name__ == "__main__":
    main()
