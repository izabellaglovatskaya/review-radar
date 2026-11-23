(function () {
  "use strict";

  function cssColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function showError() {
    document.documentElement.classList.add("charts-failed");
    document.getElementById("chart-error").hidden = false;
  }

  function lineDataset(days, key, label, color, pointStyle, borderDash) {
    return {
      label: label,
      data: days.map(function (day) { return day[key]; }),
      borderColor: color,
      backgroundColor: color,
      pointStyle: pointStyle,
      pointRadius: 4,
      borderDash: borderDash,
      borderWidth: 2,
      tension: 0
    };
  }

  function barOptions(max, suffix) {
    return {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          beginAtZero: true,
          max: max,
          ticks: {
            precision: 0,
            callback: function (value) { return value + suffix; }
          }
        }
      }
    };
  }

  function drawDaily(days) {
    var datasets = [
      lineDataset(days, "positive", "Positive", cssColor("--positive"), "circle", []),
      lineDataset(days, "neutral", "Neutral", cssColor("--neutral"), "rect", [8, 4]),
      lineDataset(days, "negative", "Negative", cssColor("--negative"), "triangle", [2, 3])
    ];
    if (days.some(function (day) { return day.not_analyzed > 0; })) {
      datasets.push(
        lineDataset(days, "not_analyzed", "Not analyzed", cssColor("--muted"), "crossRot", [12, 3, 2, 3])
      );
    }
    new Chart(document.getElementById("chart-daily"), {
      type: "line",
      data: { labels: days.map(function (day) { return day.label; }), datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { labels: { usePointStyle: true } } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
      }
    });
  }

  function drawShares(shares) {
    new Chart(document.getElementById("chart-shares"), {
      type: "bar",
      data: {
        labels: shares.map(function (item) { return item.label; }),
        datasets: [{
          label: "Share, %",
          data: shares.map(function (item) { return item.share; }),
          backgroundColor: [cssColor("--positive"), cssColor("--neutral"), cssColor("--negative")]
        }]
      },
      options: barOptions(100, "%")
    });
  }

  function drawTopics(topics) {
    new Chart(document.getElementById("chart-topics"), {
      type: "bar",
      data: {
        labels: topics.map(function (item) { return item.label; }),
        datasets: [{
          label: "Negative reviews",
          data: topics.map(function (item) { return item.count; }),
          backgroundColor: cssColor("--negative")
        }]
      },
      options: barOptions(undefined, "")
    });
  }

  function start() {
    if (!window.Chart) {
      showError();
      return;
    }
    var payload = JSON.parse(document.getElementById("dashboard-data").textContent);
    document.querySelectorAll(".chart").forEach(function (element) { element.hidden = false; });
    Chart.defaults.color = cssColor("--text");
    Chart.defaults.borderColor = cssColor("--grid");
    drawDaily(payload.days);
    drawShares(payload.shares);
    drawTopics(payload.topics);
  }

  try {
    start();
  } catch (error) {
    showError();
    throw error;
  }
})();
