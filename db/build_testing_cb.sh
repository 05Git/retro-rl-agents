#!usr/bin/env bash

# Run this in the DB directory

for f in ./migrations/*.sql; do
    sqlite3 testing_database.db < $f
done