
Files run as requested  

0) Must install requirements.

1) setup_data.py will build the SQLite DB 
    needs the folder sources to build the DB as it does in the original repo.

2) build_kg.py will build the KG
    Added some more properties to the Rule Nodes (embedding properties), is described in the report.

3) query_systems.py contains all the logic for cypher query and retrieval

4) auto_test.py version had some modifications to the functions that it imports from query_systems.py but it works as a stand alone and works with the default test_data.json for testing.