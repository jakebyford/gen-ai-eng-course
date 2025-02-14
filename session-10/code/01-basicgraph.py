# create environment
# python -m venv venv
# source venv\Scripts\activate
# pip install neo4j

# To remove everything :  MATCH(n) DETACH DELETE n

from neo4j import GraphDatabase
uri = "bolt://localhost:7687"
# url = "neo4j+s://0b96ebce.databases.neo4j.io"

user = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(user, password))

def create_data(tx):
    tx.run("CREATE (p:Person { name: 'John Doe', age: 30})")
    tx.run("CREATE (p:Person { name: 'Alice Smith', age: 28})")
    tx.run("CREATE (c:Company { name: 'Neo4J LLP', industry: 'Software'})")

    tx.run("MATCH (p:Person {name: 'John Doe'}), (c:Company {name: 'Neo4J LLP'}) CREATE (p)-[:WORKS_AT]->(c)")
    tx.run("MATCH (p1:Person {name: 'John Doe'}), (p2:Person {name: 'Alice Smith'}) CREATE (p1)-[:KNOWS]->(p2)")


def query_data(tx):
    result = tx.run("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name AS Name, c.name as Company")

    for record in result:
        print(f"{record['Name']} works at {record['Company']}")

    result = tx.run("MATCH (p1:Person)-[:KNOWS]->(p2:Person) RETURN p1.name AS Person1, p2.name as Person2")

    for record in result:
        print(f"{record['Person1']} knows {record['Person2']}")


with driver.session() as session:
    session.execute_write(create_data)
