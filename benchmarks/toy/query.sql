SELECT c.c_id, COUNT(*) AS n_orders
FROM customers AS c JOIN orders AS o ON c.c_id = o.o_cust
GROUP BY c.c_id
HAVING COUNT(*) > 2
