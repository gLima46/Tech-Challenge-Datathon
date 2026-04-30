FROM grafana/grafana:11.1.0

COPY provisioning/datasources/prometheus.yml /etc/grafana/provisioning/datasources/prometheus.yml
COPY provisioning/dashboards/dashboard.yml /etc/grafana/provisioning/dashboards/dashboard.yml
COPY dashboards/lstm-overview.json /var/lib/grafana/dashboards/lstm-overview.json
