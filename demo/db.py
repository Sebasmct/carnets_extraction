import urllib.parse

from sqlalchemy import URL
from sqlalchemy.sql import func
from sqlalchemy import (create_engine, 
                        MetaData, Table, Column, DateTime, Text, String,
                        insert)


# load it with enviroment variables in a .env file
password = urllib.parse.quote_plus("password") # plain (unescaped) text

url_object = URL.create(
    "postgresql+psycopg2",
    username="postgres",
    password=password,  
    host="localhost",
    port=5432,
    database="postgres",
)
engine = create_engine(url_object, echo=True)

metadata_obj = MetaData(schema="public")
carnet_detection_table = Table(
    "carnet_detection",
    metadata_obj,
    Column("datetime", DateTime(timezone=True), primary_key=True, server_default=func.now()),
    Column("name", Text),
    Column("role", Text),
    Column("id_number", String(10))
)
# metadata_obj.create_all(engine)



# carnet_detection_table = Table("carnet_detection", metadata_obj, autoload_with=engine)

stmt = insert(carnet_detection_table).values(name="Sebastian",
                                             role="Coordinador GPS",
                                             id_number=10)

with engine.begin() as conn:
    result = conn.execute(stmt)