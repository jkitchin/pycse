<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta content="text/html; charset=utf-8" http-equiv="content-type">
<div id='main'>
    <h2>Cork - Administration page</h2>
    <p>Welcome {{current_user.username}}, your role is: {{current_user.role}},
    access time: {{current_user.session_accessed_time}}</p>
    <div id='commands'>
      <p>Create new user:</p>
      <form action="create_user" method="post">
          <p><label>Username</label> <input type="text" name="username" /></p>
          <p><label>Role</label> <input type="text" name="role" /></p>
          <p><label>Password</label> <input type="password" name="password" /></p>
          <button type="submit" > OK </button>
          <button type="button" class="close"> Cancel </button>
      </form>
      <br />
      <p>Delete user:</p>
      <form action="delete_user" method="post">
          <p><label>Username</label> <input type="text" name="username" /></p>
          <button type="submit" > OK </button>
          <button type="button" class="close"> Cancel </button>
      </form>
      <br />
      <p>Create new role:</p>
      <form action="create_role" method="post">
          <p><label>Role</label> <input type="text" name="role" /></p>
          <p><label>Level</label> <input type="text" name="level" /></p>
          <button type="submit" > OK </button>
          <button type="button" class="close"> Cancel </button>
      </form>
      <br />
      <p>Delete role:</p>
      <form action="delete_role" method="post">
          <p><label>Role</label> <input type="text" name="role" /></p>
          <button type="submit" > OK </button>
          <button type="button" class="close"> Cancel </button>
      </form>
    </div>
    <div id="users">
        <table>
            <tr><th>Username</th><th>Role</th><th>Email</th><th>Description</th></tr>
            %for u in users:
            <tr><td>{{u[0]}}</td><td>{{u[1]}}</td><td>{{u[2]}}</td><td>{{u[2]}}</td></tr>
            %end
        </table>
        <br/>
        <table>
            <tr><th>Role</th><th>Level</th></tr>
            %for r in roles:
            <tr><td>{{r[0]}}</td><td>{{r[1]}}</td></tr>
            %end
        </table>
        <p>(Reload page to refresh)</p>
    </div>

    <div class="clear"></div>

    <div id='status'><p>Ready.</p></div>
    <div id="urls">
      <a href="/">index</a> <a href="/logout">logout</a>
    </div>
    <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
    <script>
        // Prevent form submission, send POST asynchronously and parse returned JSON
        $('form').submit(function() {
            $("div#status").fadeIn(100);
            z = $(this);
            $.post($(this).attr('action'), $(this).serialize(), function(j){
              if (j.ok) {
                $("div#status").css("background-color", "#f0fff0");
                $("div#status p").text('Ok.');
              } else {
                $("div#status").css("background-color", "#fff0f0");
                $("div#status p").text(j.msg);
              }
              $("div#status").delay(800).fadeOut(500);
            }, "json");
            return false;
        });
    </script>
</div>
<style>
div#commands { width: 45%%; float: left}
div#users { width: 45%; float: right}
div#main {
    color: #777;
    margin: auto;
    margin-left: 5em;
    font-size: 80%;
}
input {
    background: #f8f8f8;
    border: 1px solid #777;
    margin: auto;
}
input:hover { background: #fefefe}
label {
  width: 8em;
  float: left;
  text-align: right;
  margin-right: 0.5em;
  display: block
}
button {
    margin-left: 13em;
}
button.close {
    margin-left: .1em;
}
div#status {
    border: 1px solid #999;
    padding: .5em;
    margin: 2em;
    width: 15em;
    -moz-border-radius: 10px;
    border-radius: 10px;
}
.clear { clear: both;}
div#urls {
  position:absolute;
  top:0;
  right:1em;
}
</style>

