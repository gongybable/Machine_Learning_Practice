FROM python:3.7

RUN pip install numpy pandas matplotlib jupyter seaborn scipy sklearn

CMD ["jupyter", "notebook", "/opt/capstone.ipynb", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]