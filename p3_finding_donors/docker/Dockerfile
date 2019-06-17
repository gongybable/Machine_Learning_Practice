FROM python:3.7

RUN pip install numpy pandas matplotlib jupyter sklearn

CMD ["jupyter", "notebook", "/opt/finding_donors.ipynb", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]