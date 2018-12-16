(function () {
    'use strict';
    // two way data binding(to UI)
    var vm = new Vue({
        el: '#app',

        data: {
            newItem: '',
            todos: [
                
            ]
        },
        methods: {
            // addItem: function(e) {
            //   e.preventDefault();
            //   this.todos.push(this.newItem);
            // }
            addItem: function () {
                this.todos.push(this.newItem);
                this.newItem = '';
            },
            deleteItem: function (index) {
                if (confirm('Are you sure?')) {
                    this.todos.splice(index, 1);
                }
            }
        }
    });
})();