FROM python:3.7

RUN pip install numpy pandas matplotlib jupyter

CMD ["jupyter", "notebook", "/opt/titanic_survival_exploration.ipynb", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]