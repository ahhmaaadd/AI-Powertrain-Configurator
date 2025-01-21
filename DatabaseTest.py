from neo4j import GraphDatabase

uri = "neo4j+ssc://5a1678ee.databases.neo4j.io"
#uri = "neo4j+ssc://aabb928f.databases.neo4j.io"
username = "neo4j"
password = "ATFhtDuFfTLI17N_1WehNl1wm5efbXMDDOsyQsAeRAU"
#password = "LNXHVHvGij-h52MVm0xTPJtJKZ01ltXZ75JhYS7Wvf8"

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    with driver.session() as session:
        result = session.run("RETURN 'Connection successful!' AS message")
        print(result.single()["message"])
except Exception as e:
    print(f"Connection failed: {e}")
finally:
    driver.close()
