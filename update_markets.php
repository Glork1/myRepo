<?php

$host_name = 'db703376053.db.1and1.com';
$database = 'db703376053';
$user_name = 'dbo703376053';
$password = '';


$connect = mysqli_connect($host_name, $user_name, $password, $database);
if (mysqli_errno()) {
    mail('nika.shishonkova@gmail.com','update_markets on Bittrex: Failed to connect to MySQL: '.mysqli_error(),phpversion());
} else {
	$url = "https://bittrex.com/api/v1.1/public/getmarkets";

	$json = json_decode(file_get_contents($url));
	
	foreach ($json->{'result'} as $json_row )
	{
    	$id = $json_row->{'MarketName'};
    	$is_active = $json_row->{'IsActive'} ? '1' : '0';
    	
    	$select_query = 'SELECT IsActive FROM `markets` WHERE `MarketName` = "'.$id.'";';
		$select_result = $connect->query($select_query);
    	
    	$query = '';
    	if ($select_result->num_rows > 0 && $select_result->fetch_assoc()['IsActive'] == $json_row->{'IsActive'}) 
    	{
        	continue;
        }
        // row exists but need to be updated
        else if ($select_result->num_rows > 0)
        {
	        $query = 'UPDATE `'.$database.'`.`markets` SET `IsActive` = "'.$json_row->{'IsActive'}.'" WHERE `MarketName` = "'.$id.'";';
        }
        // otherwise insert new row
        else
        {
	       $query = 'INSERT INTO `'.$database.'`.`markets` (`id`, `MarketCurrency`, `BaseCurrency`, `MarketCurrencyLong`, `BaseCurrencyLong`, `MinTradeSize`, `MarketName`, `IsActive`, `Created`, `Notice`, `IsSponsored`) VALUES (NULL, \''.$json_row->{'MarketCurrency'}.'\', \''.$json_row->{'BaseCurrency'}.'\', \''.$json_row->{'MarketCurrencyLong'}.'\', \''.$json_row->{'BaseCurrencyLong'}.'\', \''.$json_row->{'MinTradeSize'}.'\', \''.$json_row->{'MarketName'}.'\', \''.$json_row->{'IsActive'}.'\', \''.$json_row->{'Created'}.'\', '.($json_row->{'Notice'}=='' ? 'NULL' : '\''.$json_row->{'Notice'}.'\'').', '.($json_row->{'IsSponsored'}=='' ? 'NULL' : '\''.$json_row->{'IsSponsored'}.'\'').');';
        }
    	//echo '<div>'.$id.': '.$query.'</div>';
    	
    	echo $connect->query($query);
	}

    mysqli_close($connect);
}
?>

<?php
$host_name = 'db703688159.db.1and1.com';
$database = 'db703688159';
$user_name = 'dbo703688159';
$password = 'k6jJ60ls';

$connect = mysqli_connect($host_name, $user_name, $password, $database);
if (mysqli_errno()) {
    mail('nika.shishonkova@gmail.com','update_markets on Poloniex: Failed to connect to MySQL: '.mysqli_error(),phpversion());
} else {
	$url = "https://poloniex.com/public?command=returnCurrencies";

	$json = json_decode(file_get_contents($url), true);
	
	foreach ($json as $key => $value )
	{
    	$id = $value['id'];
    	//echo "<div>".sizeof($json_row).":".$id."</div>";
    	
    	$select_query = 'SELECT `id`, `disabled`, `delisted`, `frozen` FROM `currencies` WHERE `key` = "'.$key.'";';
    	
    	//echo '<div>'.$select_query.'</div>';
		$select_result = $connect->query($select_query);
    	
    	$query = '';
    	if ($select_result->num_rows > 0) 
    	{
	    	$res = $select_result->fetch_assoc();
	    	
	    	if ($res['id'] == $value['id'] && $res['disabled'] == $value['disabled'] && $res['delisted'] == $value['delisted'] && $res['frozen'] == $value['frozen'])
	        	continue;
	        // row exists but need to be updated
	        $query = 'UPDATE `'.$database.'`.`currencies` SET `disabled` = "'.$value['disabled'].'", `delisted` = "'.$value['delisted'].'",`frozen` = "'.$value['frozen'].'" WHERE `key` = "'.$key.'";';
        }
        // otherwise insert new row
        else
        {
	        $query = 'INSERT INTO `db703688159`.`currencies` (`id`, `key`, `name`, `txFee`, `minConf`, `disabled`, `delisted`, `frozen`) VALUES ("'.$value['id'].'", "'.$key.'", "'.$value['name'].'", "'.$value['txFee'].'", "'.$value['minConf'].'", "'.$value['disabled'].'", "'.$value['delisted'].'", "'.$value['frozen'].'");';
        }
    	//echo '<div>'.$query.'</div>';
    	
    	echo $connect->query($query);
	}

    mysqli_close($connect);
}
?>

<?php
$host_name = 'db703688159.db.1and1.com';
$database = 'db703688159';
$user_name = 'dbo703688159';
$password = 'k6jJ60ls';

$connect = mysqli_connect($host_name, $user_name, $password, $database);
if (mysqli_errno()) {
    mail('nika.shishonkova@gmail.com','update_markets on Poloniex: Failed to connect to MySQL: '.mysqli_error(),phpversion());
} else {
	$url = "https://poloniex.com/public?command=returnTicker";

	$json = json_decode(file_get_contents($url), true);
	
	foreach ($json as $key => $value )
	{
    	$id = $value['id'];
    	//echo "<div>".sizeof($json_row).":".$id."</div>";
    	
    	$select_query = 'SELECT `id`, `isFrozen` FROM `tickers` WHERE `ccy` = "'.$key.'";';
    	
    	//echo '<div>'.$select_query.'</div>';
		$select_result = $connect->query($select_query);
    	
    	$query = '';
    	if ($select_result->num_rows > 0) 
    	{
	    	$res = $select_result->fetch_assoc();
	    	if ($res['id'] == $value['id'] && $res['isFrozen'] == $value['isFrozen'])
	        	continue;
	        // row exists but need to be updated
	        $query = 'UPDATE `'.$database.'`.`tickers` SET `isFrozen` = "'.$value['isFrozen'].'" WHERE `ccy` = "'.$key.'";';
        }
        // otherwise insert new row
        else
        {
	        $query = 'INSERT INTO `'.$database.'`.`tickers` (`id`, `ccy`, `last`, `lowestAsk`, `highestBid`, `percentChange`, `baseVolume`, `quoteVolume`, `isFrozen`, `high24hr`, `low24hr`) VALUES ("'.$value['id'].'", "'.$key.'", "'.$value['last'].'", "'.$value['lowestAsk'].'", "'.$value['highestBid'].'", "'.$value['percentChange'].'", "'.$value['baseVolume'].'", "'.$value['quoteVolume'].'", "'.$value['isFrozen'].'", "'.$value['high24hr'].'", "'.$value['low24hr'].'");';
        }
    	//echo '<div>'.$query.'</div>';
    	
    	echo $connect->query($query);
	}
	
	$base_ccy_query = "SELECT t.id as 'id', t.ccy, c_base.id as 'base_ccy_id', c2.id as 'ccy_id' FROM `tickers` t JOIN currencies c_base ON t.ccy LIKE CONCAT(c_base.key, '%') JOIN currencies c2 ON t.ccy LIKE CONCAT('%', c2.key) WHERE CONCAT(c_base.key,'_',c2.key) = t.ccy AND t.isFrozen = 0 AND (t.base_ccy_id IS NULL OR t.ccy_id IS NULL)";
	$base_ccy_result = $connect->query($base_ccy_query);
	while($row = $base_ccy_result->fetch_assoc()) {
    	$base_ccy_update_query = "UPDATE `".$database."`.`tickers` SET `base_ccy_id` = '".$row['base_ccy_id']."', `ccy_id` = '".$row['ccy_id']."' WHERE `tickers`.`id` = ".$row['id'].";";
    	echo $connect->query($base_ccy_update_query);
    }
	
    mysqli_close($connect);
}
?>