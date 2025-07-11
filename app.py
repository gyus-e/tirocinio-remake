from flask import Flask
import routes

app = Flask(__name__)

app.register_blueprint(routes.configurations_blueprint)
app.register_blueprint(routes.cag_blueprint, url_prefix="/configurations")

app.run(host="0.0.0.0", port=5001, debug=True)