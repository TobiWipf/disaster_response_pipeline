<div class="container">
    {% block content %}
        <div class="page-header">
            <h1 class="text-center">Overview of Training Dataset</h1>
        </div>
    {% endblock %}

    {% for id in ids %}
        <div id="{{id}}"></div>
    {% endfor %}
</div>

{% block message %}
        {% endblock %}
        
        <script type="text/javascript">
    const graphs = {{graphJSON | safe}};
    const ids = {{ids | safe}};
    for(let i in graphs) {
        Plotly.plot(ids[i], graphs[i].data, graphs[i].layout);
    }
</script>