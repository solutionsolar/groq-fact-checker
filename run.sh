#!/bin/bash
poetry install
poetry run flask db upgrade
poetry run python app.py