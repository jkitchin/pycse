<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta content="text/html; charset=utf-8" http-equiv="content-type">
<div class="box">
    <h2>Password change</h2>
    <p>Please insert your new password:</p>
    <form action="/change_password" method="post">
        <input type="password" name="password" />
        <input type="hidden" name="reset_code" value="{{reset_code}}" />

        <br/><br/>
        <button type="submit" > OK </button>
        <button type="button" class="close"> Cancel </button>
    </form>
    <br />
</div>
<style>
div {
    color: #777;
    margin: auto;
    width: 20em;
    text-align: center;
}
input {
    background: #f8f8f8;
    border: 1px solid #777;
    margin: auto;
}
input:hover { background: #fefefe}
</style>
