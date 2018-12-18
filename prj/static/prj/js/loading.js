function openData() {
    window.open("{% url 'myapp:User_View' %}");
}

setTimeout("openData()", 3000);