from flask import Flask
import cloudinary



app = Flask(__name__)



#app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql://root:Rizalmohamad123@localhost/rest_flask'

cloud = cloudinary.config(
    cloud_name="dk4dgvu4w",
    api_key="312482332544282",
    api_secret="1oSO-d9c8he7Z7Lb9CjTNjPFMmk"
)
from app import routes

